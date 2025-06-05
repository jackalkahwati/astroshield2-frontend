"""
Subsystem 3: Command & Control / Logistics
Battle management command and control for space operations
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import json

from ..kafka.kafka_client import (
    WeldersArcKafkaClient,
    WeldersArcMessage,
    KafkaTopics,
    SubsystemID,
    EventType
)
from .ss1_target_modeling import TargetModel, BehaviorPattern, ObjectType
from .ss2_state_estimation import StateVector

logger = logging.getLogger(__name__)


class CommandType(Enum):
    """Types of commands"""
    COLLECTION_TASK = "collection_task"
    MANEUVER_TASK = "maneuver_task"
    MODE_CHANGE = "mode_change"
    ALERT_RESPONSE = "alert_response"
    DEFENSIVE_ACTION = "defensive_action"
    SENSOR_TASKING = "sensor_tasking"
    WEAPON_RELEASE = "weapon_release"
    STAND_DOWN = "stand_down"


class CommandPriority(Enum):
    """Command priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    ROUTINE = 5


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


class AssetType(Enum):
    """Types of controllable assets"""
    SENSOR = "sensor"
    SATELLITE = "satellite"
    GROUND_STATION = "ground_station"
    DEFENSIVE_SYSTEM = "defensive_system"
    COMMUNICATIONS = "communications"


@dataclass
class Asset:
    """Controllable asset"""
    asset_id: str
    asset_type: AssetType
    capabilities: List[str]
    location: Dict[str, float]  # lat, lon, alt or orbit params
    status: str
    availability: float  # 0-1
    current_task: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Command:
    """Command to be executed"""
    command_id: str
    command_type: CommandType
    priority: CommandPriority
    target_asset: str
    parameters: Dict[str, Any]
    issuer: str
    issued_time: datetime
    execution_time: datetime
    expiry_time: datetime
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class TaskPlan:
    """Coordinated task plan"""
    plan_id: str
    objective: str
    commands: List[Command]
    start_time: datetime
    end_time: datetime
    assets_involved: List[str]
    priority: CommandPriority
    status: TaskStatus = TaskStatus.PENDING
    metrics: Dict[str, Any] = field(default_factory=dict)


class TaskScheduler:
    """Schedule and optimize task execution"""
    
    def __init__(self):
        self.schedule_queue: deque = deque()
        self.executing_tasks: Dict[str, Command] = {}
        self.resource_availability: Dict[str, float] = {}
        
    async def schedule_command(self, command: Command, assets: Dict[str, Asset]) -> bool:
        """Schedule a command for execution"""
        # Check asset availability
        target_asset = assets.get(command.target_asset)
        if not target_asset:
            logger.error(f"Asset {command.target_asset} not found")
            return False
            
        if target_asset.availability < 0.5:
            logger.warning(f"Asset {command.target_asset} has low availability: {target_asset.availability}")
            
        # Check for conflicts
        conflicts = await self._check_conflicts(command, target_asset)
        if conflicts:
            # Try to reschedule
            new_time = await self._find_available_slot(command, target_asset)
            if new_time:
                command.execution_time = new_time
            else:
                return False
                
        # Add to schedule
        self._insert_by_priority(command)
        return True
        
    async def _check_conflicts(self, command: Command, asset: Asset) -> List[Command]:
        """Check for scheduling conflicts"""
        conflicts = []
        
        for scheduled in self.schedule_queue:
            if scheduled.target_asset == command.target_asset:
                # Check time overlap
                if self._times_overlap(command, scheduled):
                    conflicts.append(scheduled)
                    
        return conflicts
        
    def _times_overlap(self, cmd1: Command, cmd2: Command) -> bool:
        """Check if two commands have overlapping execution times"""
        # Estimate execution duration based on command type
        duration1 = self._estimate_duration(cmd1)
        duration2 = self._estimate_duration(cmd2)
        
        end1 = cmd1.execution_time + duration1
        end2 = cmd2.execution_time + duration2
        
        return not (end1 <= cmd2.execution_time or cmd1.execution_time >= end2)
        
    def _estimate_duration(self, command: Command) -> timedelta:
        """Estimate command execution duration"""
        durations = {
            CommandType.COLLECTION_TASK: timedelta(minutes=10),
            CommandType.MANEUVER_TASK: timedelta(minutes=30),
            CommandType.MODE_CHANGE: timedelta(minutes=5),
            CommandType.ALERT_RESPONSE: timedelta(minutes=2),
            CommandType.DEFENSIVE_ACTION: timedelta(minutes=15),
            CommandType.SENSOR_TASKING: timedelta(minutes=20),
            CommandType.WEAPON_RELEASE: timedelta(minutes=1),
            CommandType.STAND_DOWN: timedelta(minutes=1)
        }
        
        return durations.get(command.command_type, timedelta(minutes=10))
        
    async def _find_available_slot(self, command: Command, asset: Asset) -> Optional[datetime]:
        """Find next available time slot for command"""
        current_time = datetime.utcnow()
        test_time = max(current_time, command.execution_time)
        
        # Try to find slot within next 24 hours
        max_time = current_time + timedelta(hours=24)
        
        while test_time < max_time:
            # Create test command with new time
            test_cmd = Command(
                command_id=command.command_id,
                command_type=command.command_type,
                priority=command.priority,
                target_asset=command.target_asset,
                parameters=command.parameters,
                issuer=command.issuer,
                issued_time=command.issued_time,
                execution_time=test_time,
                expiry_time=command.expiry_time
            )
            
            conflicts = await self._check_conflicts(test_cmd, asset)
            if not conflicts:
                return test_time
                
            # Try 5 minutes later
            test_time += timedelta(minutes=5)
            
        return None
        
    def _insert_by_priority(self, command: Command):
        """Insert command in queue by priority and time"""
        # Find insertion point
        insert_idx = 0
        for i, cmd in enumerate(self.schedule_queue):
            if command.priority.value < cmd.priority.value:
                insert_idx = i
                break
            elif (command.priority == cmd.priority and 
                  command.execution_time < cmd.execution_time):
                insert_idx = i
                break
            else:
                insert_idx = i + 1
                
        self.schedule_queue.insert(insert_idx, command)
        
    async def get_next_command(self) -> Optional[Command]:
        """Get next command to execute"""
        current_time = datetime.utcnow()
        
        while self.schedule_queue:
            command = self.schedule_queue[0]
            
            # Check if expired
            if command.expiry_time < current_time:
                self.schedule_queue.popleft()
                command.status = TaskStatus.CANCELLED
                continue
                
            # Check if ready to execute
            if command.execution_time <= current_time:
                return self.schedule_queue.popleft()
                
            break
            
        return None


class BattleManager:
    """Manage battle rhythm and coordination"""
    
    def __init__(self):
        self.threat_responses: Dict[str, List[Command]] = {}
        self.defensive_postures = {
            "nominal": self._nominal_posture,
            "elevated": self._elevated_posture,
            "tactical": self._tactical_posture,
            "strategic": self._strategic_posture
        }
        self.current_posture = "nominal"
        
    async def assess_situation(self, threats: List[TargetModel]) -> str:
        """Assess overall situation and recommend posture"""
        if not threats:
            return "nominal"
            
        # Count threat levels
        high_threats = sum(1 for t in threats if t.anomaly_score > 0.8)
        medium_threats = sum(1 for t in threats if 0.5 < t.anomaly_score <= 0.8)
        
        # Check for specific aggressive behaviors
        aggressive = sum(1 for t in threats if t.behavior_pattern == BehaviorPattern.AGGRESSIVE)
        
        if aggressive > 0 or high_threats > 2:
            return "strategic"
        elif high_threats > 0:
            return "tactical"
        elif medium_threats > 3:
            return "elevated"
        else:
            return "nominal"
            
    async def generate_response_plan(
        self,
        threat: TargetModel,
        assets: Dict[str, Asset]
    ) -> List[Command]:
        """Generate response plan for threat"""
        commands = []
        
        # Prioritize by threat level
        if threat.anomaly_score > 0.9:
            priority = CommandPriority.CRITICAL
        elif threat.anomaly_score > 0.7:
            priority = CommandPriority.HIGH
        else:
            priority = CommandPriority.MEDIUM
            
        # 1. Enhanced tracking
        tracking_cmd = await self._generate_tracking_command(threat, assets, priority)
        if tracking_cmd:
            commands.append(tracking_cmd)
            
        # 2. Sensor collection
        collection_cmds = await self._generate_collection_commands(threat, assets, priority)
        commands.extend(collection_cmds)
        
        # 3. Defensive measures if needed
        if threat.behavior_pattern in [BehaviorPattern.AGGRESSIVE, BehaviorPattern.EVASIVE]:
            defensive_cmds = await self._generate_defensive_commands(threat, assets, priority)
            commands.extend(defensive_cmds)
            
        # 4. Alert generation
        alert_cmd = self._generate_alert_command(threat, priority)
        commands.append(alert_cmd)
        
        return commands
        
    async def _generate_tracking_command(
        self,
        threat: TargetModel,
        assets: Dict[str, Asset],
        priority: CommandPriority
    ) -> Optional[Command]:
        """Generate enhanced tracking command"""
        # Find best sensor for tracking
        best_sensor = None
        best_score = 0
        
        for asset in assets.values():
            if asset.asset_type == AssetType.SENSOR and "tracking" in asset.capabilities:
                # Score based on availability and location
                score = asset.availability
                if best_score < score:
                    best_sensor = asset
                    best_score = score
                    
        if best_sensor:
            return Command(
                command_id=f"track-{threat.object_id}-{datetime.utcnow().timestamp()}",
                command_type=CommandType.SENSOR_TASKING,
                priority=priority,
                target_asset=best_sensor.asset_id,
                parameters={
                    "mode": "enhanced_tracking",
                    "target_id": threat.object_id,
                    "track_rate": "high",
                    "duration_minutes": 60
                },
                issuer="battle_manager",
                issued_time=datetime.utcnow(),
                execution_time=datetime.utcnow(),
                expiry_time=datetime.utcnow() + timedelta(hours=1)
            )
            
        return None
        
    async def _generate_collection_commands(
        self,
        threat: TargetModel,
        assets: Dict[str, Asset],
        priority: CommandPriority
    ) -> List[Command]:
        """Generate sensor collection commands"""
        commands = []
        
        # Find available sensors
        for asset in assets.values():
            if (asset.asset_type == AssetType.SENSOR and 
                asset.availability > 0.3 and
                not asset.current_task):
                
                # Determine collection type based on sensor capabilities
                collection_type = "optical"
                if "radar" in asset.capabilities:
                    collection_type = "radar"
                elif "rf" in asset.capabilities:
                    collection_type = "rf"
                    
                cmd = Command(
                    command_id=f"collect-{threat.object_id}-{asset.asset_id}-{datetime.utcnow().timestamp()}",
                    command_type=CommandType.COLLECTION_TASK,
                    priority=priority,
                    target_asset=asset.asset_id,
                    parameters={
                        "target_id": threat.object_id,
                        "collection_type": collection_type,
                        "priority": priority.value,
                        "requirements": {
                            "quality": "high",
                            "coverage": "continuous"
                        }
                    },
                    issuer="battle_manager",
                    issued_time=datetime.utcnow(),
                    execution_time=datetime.utcnow() + timedelta(minutes=5),
                    expiry_time=datetime.utcnow() + timedelta(hours=2)
                )
                
                commands.append(cmd)
                
                # Limit to 3 sensors per threat
                if len(commands) >= 3:
                    break
                    
        return commands
        
    async def _generate_defensive_commands(
        self,
        threat: TargetModel,
        assets: Dict[str, Asset],
        priority: CommandPriority
    ) -> List[Command]:
        """Generate defensive action commands"""
        commands = []
        
        # Find defensive assets
        for asset in assets.values():
            if asset.asset_type == AssetType.DEFENSIVE_SYSTEM:
                cmd = Command(
                    command_id=f"defend-{threat.object_id}-{asset.asset_id}-{datetime.utcnow().timestamp()}",
                    command_type=CommandType.DEFENSIVE_ACTION,
                    priority=CommandPriority.HIGH,  # Always high for defensive
                    target_asset=asset.asset_id,
                    parameters={
                        "threat_id": threat.object_id,
                        "threat_level": threat.anomaly_score,
                        "mode": "active_defense",
                        "rules_of_engagement": "defensive_only"
                    },
                    issuer="battle_manager",
                    issued_time=datetime.utcnow(),
                    execution_time=datetime.utcnow(),
                    expiry_time=datetime.utcnow() + timedelta(hours=4)
                )
                
                commands.append(cmd)
                
        # Add maneuver commands for friendly satellites
        if threat.behavior_pattern == BehaviorPattern.AGGRESSIVE:
            for asset in assets.values():
                if (asset.asset_type == AssetType.SATELLITE and 
                    "maneuver" in asset.capabilities):
                    
                    cmd = Command(
                        command_id=f"evade-{asset.asset_id}-{datetime.utcnow().timestamp()}",
                        command_type=CommandType.MANEUVER_TASK,
                        priority=CommandPriority.HIGH,
                        target_asset=asset.asset_id,
                        parameters={
                            "maneuver_type": "evasive",
                            "threat_id": threat.object_id,
                            "delta_v_authorized": 50  # m/s
                        },
                        issuer="battle_manager",
                        issued_time=datetime.utcnow(),
                        execution_time=datetime.utcnow() + timedelta(minutes=10),
                        expiry_time=datetime.utcnow() + timedelta(hours=1)
                    )
                    
                    commands.append(cmd)
                    
        return commands
        
    def _generate_alert_command(self, threat: TargetModel, priority: CommandPriority) -> Command:
        """Generate alert command"""
        return Command(
            command_id=f"alert-{threat.object_id}-{datetime.utcnow().timestamp()}",
            command_type=CommandType.ALERT_RESPONSE,
            priority=priority,
            target_asset="command_center",
            parameters={
                "threat_id": threat.object_id,
                "threat_type": threat.behavior_pattern.value,
                "anomaly_score": threat.anomaly_score,
                "recommended_actions": ["enhance_tracking", "prepare_defenses"],
                "alert_level": priority.value
            },
            issuer="battle_manager",
            issued_time=datetime.utcnow(),
            execution_time=datetime.utcnow(),
            expiry_time=datetime.utcnow() + timedelta(hours=24)
        )
        
    async def _nominal_posture(self, assets: Dict[str, Asset]) -> List[Command]:
        """Nominal defensive posture"""
        commands = []
        
        for asset in assets.values():
            if asset.asset_type == AssetType.SENSOR:
                # Normal scanning pattern
                cmd = Command(
                    command_id=f"scan-{asset.asset_id}-{datetime.utcnow().timestamp()}",
                    command_type=CommandType.MODE_CHANGE,
                    priority=CommandPriority.ROUTINE,
                    target_asset=asset.asset_id,
                    parameters={
                        "mode": "survey",
                        "scan_pattern": "standard"
                    },
                    issuer="battle_manager",
                    issued_time=datetime.utcnow(),
                    execution_time=datetime.utcnow() + timedelta(minutes=15),
                    expiry_time=datetime.utcnow() + timedelta(hours=8)
                )
                commands.append(cmd)
                
        return commands
        
    async def _elevated_posture(self, assets: Dict[str, Asset]) -> List[Command]:
        """Elevated defensive posture"""
        commands = await self._nominal_posture(assets)
        
        # Increase sensor sensitivity
        for asset in assets.values():
            if asset.asset_type == AssetType.SENSOR:
                cmd = Command(
                    command_id=f"enhance-{asset.asset_id}-{datetime.utcnow().timestamp()}",
                    command_type=CommandType.MODE_CHANGE,
                    priority=CommandPriority.MEDIUM,
                    target_asset=asset.asset_id,
                    parameters={
                        "mode": "enhanced_detection",
                        "sensitivity": "high"
                    },
                    issuer="battle_manager",
                    issued_time=datetime.utcnow(),
                    execution_time=datetime.utcnow(),
                    expiry_time=datetime.utcnow() + timedelta(hours=4)
                )
                commands.append(cmd)
                
        return commands
        
    async def _tactical_posture(self, assets: Dict[str, Asset]) -> List[Command]:
        """Tactical defensive posture"""
        commands = []
        
        # All sensors to tactical mode
        for asset in assets.values():
            if asset.asset_type == AssetType.SENSOR:
                cmd = Command(
                    command_id=f"tactical-{asset.asset_id}-{datetime.utcnow().timestamp()}",
                    command_type=CommandType.MODE_CHANGE,
                    priority=CommandPriority.HIGH,
                    target_asset=asset.asset_id,
                    parameters={
                        "mode": "tactical",
                        "focus": "threat_detection",
                        "reporting": "immediate"
                    },
                    issuer="battle_manager",
                    issued_time=datetime.utcnow(),
                    execution_time=datetime.utcnow(),
                    expiry_time=datetime.utcnow() + timedelta(hours=2)
                )
                commands.append(cmd)
                
            elif asset.asset_type == AssetType.DEFENSIVE_SYSTEM:
                # Activate defensive systems
                cmd = Command(
                    command_id=f"activate-{asset.asset_id}-{datetime.utcnow().timestamp()}",
                    command_type=CommandType.MODE_CHANGE,
                    priority=CommandPriority.HIGH,
                    target_asset=asset.asset_id,
                    parameters={
                        "mode": "active",
                        "readiness": "immediate"
                    },
                    issuer="battle_manager",
                    issued_time=datetime.utcnow(),
                    execution_time=datetime.utcnow(),
                    expiry_time=datetime.utcnow() + timedelta(hours=2)
                )
                commands.append(cmd)
                
        return commands
        
    async def _strategic_posture(self, assets: Dict[str, Asset]) -> List[Command]:
        """Strategic defensive posture - highest alert"""
        commands = await self._tactical_posture(assets)
        
        # Additional strategic measures
        for asset in assets.values():
            if asset.asset_type == AssetType.SATELLITE and "maneuver" in asset.capabilities:
                # Prepare for defensive maneuvers
                cmd = Command(
                    command_id=f"prepare-{asset.asset_id}-{datetime.utcnow().timestamp()}",
                    command_type=CommandType.MODE_CHANGE,
                    priority=CommandPriority.CRITICAL,
                    target_asset=asset.asset_id,
                    parameters={
                        "mode": "defensive_ready",
                        "fuel_reserve": "combat",
                        "maneuver_authority": "full"
                    },
                    issuer="battle_manager",
                    issued_time=datetime.utcnow(),
                    execution_time=datetime.utcnow(),
                    expiry_time=datetime.utcnow() + timedelta(hours=1)
                )
                commands.append(cmd)
                
        return commands


class CommandControlSubsystem:
    """Main command and control subsystem"""
    
    def __init__(self, kafka_client: WeldersArcKafkaClient):
        self.kafka_client = kafka_client
        self.assets: Dict[str, Asset] = {}
        self.task_scheduler = TaskScheduler()
        self.battle_manager = BattleManager()
        self.active_plans: Dict[str, TaskPlan] = {}
        self.command_history: deque = deque(maxlen=10000)
        self.metrics = {
            "commands_issued": 0,
            "commands_completed": 0,
            "commands_failed": 0,
            "average_response_time": 0
        }
        
    async def initialize(self):
        """Initialize command and control subsystem"""
        # Subscribe to relevant topics
        self.kafka_client.subscribe(
            KafkaTopics.ALERT_THREAT,
            self._handle_threat_alert
        )
        
        self.kafka_client.subscribe(
            KafkaTopics.COMMAND_STATUS,
            self._handle_command_status
        )
        
        self.kafka_client.subscribe(
            KafkaTopics.ASSET_STATUS,
            self._handle_asset_status
        )
        
        # Start background tasks
        asyncio.create_task(self._command_executor())
        asyncio.create_task(self._plan_monitor())
        asyncio.create_task(self._metrics_reporter())
        
        logger.info("Command and control subsystem initialized")
        
    async def _handle_threat_alert(self, message: WeldersArcMessage):
        """Handle incoming threat alert"""
        try:
            threat_data = message.data
            
            # Create simplified threat model
            threat = TargetModel(
                object_id=threat_data["object_id"],
                object_type=ObjectType.UNKNOWN,
                last_updated=datetime.utcnow(),
                mass=1000,
                cross_sectional_area=10,
                ballistic_coefficient=100,
                area_to_mass_ratio=0.01,
                mean_motion=15,
                eccentricity=0.001,
                inclination=0,
                semi_major_axis=6778,
                behavior_pattern=BehaviorPattern(threat_data.get("behavior", "anomalous")),
                pattern_confidence=threat_data.get("confidence", 0.5),
                maneuver_capability=None,
                anomaly_score=threat_data.get("threat_level", 0.5)
            )
            
            # Generate response plan
            response_commands = await self.battle_manager.generate_response_plan(
                threat,
                self.assets
            )
            
            # Create task plan
            plan = TaskPlan(
                plan_id=f"response-{threat.object_id}-{datetime.utcnow().timestamp()}",
                objective=f"Respond to threat from {threat.object_id}",
                commands=response_commands,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow() + timedelta(hours=4),
                assets_involved=[cmd.target_asset for cmd in response_commands],
                priority=CommandPriority.HIGH
            )
            
            # Execute plan
            await self.execute_plan(plan)
            
        except Exception as e:
            logger.error(f"Error handling threat alert: {e}")
            
    async def _handle_command_status(self, message: WeldersArcMessage):
        """Handle command status update"""
        command_id = message.data["command_id"]
        status = TaskStatus(message.data["status"])
        
        # Update command in history
        for cmd in self.command_history:
            if cmd.command_id == command_id:
                cmd.status = status
                cmd.result = message.data.get("result")
                break
                
        # Update metrics
        if status == TaskStatus.COMPLETED:
            self.metrics["commands_completed"] += 1
        elif status == TaskStatus.FAILED:
            self.metrics["commands_failed"] += 1
            
        # Update asset availability
        if "asset_id" in message.data:
            asset = self.assets.get(message.data["asset_id"])
            if asset:
                if status == TaskStatus.EXECUTING:
                    asset.current_task = command_id
                    asset.availability *= 0.5
                elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    asset.current_task = None
                    asset.availability = min(1.0, asset.availability * 2)
                    
    async def _handle_asset_status(self, message: WeldersArcMessage):
        """Handle asset status update"""
        asset_data = message.data
        asset_id = asset_data["asset_id"]
        
        if asset_id in self.assets:
            # Update existing asset
            asset = self.assets[asset_id]
            asset.status = asset_data.get("status", asset.status)
            asset.availability = asset_data.get("availability", asset.availability)
            asset.location = asset_data.get("location", asset.location)
        else:
            # Create new asset
            asset = Asset(
                asset_id=asset_id,
                asset_type=AssetType(asset_data["asset_type"]),
                capabilities=asset_data.get("capabilities", []),
                location=asset_data.get("location", {}),
                status=asset_data.get("status", "unknown"),
                availability=asset_data.get("availability", 1.0),
                metadata=asset_data.get("metadata", {})
            )
            self.assets[asset_id] = asset
            
    async def _command_executor(self):
        """Execute scheduled commands"""
        while True:
            try:
                # Get next command
                command = await self.task_scheduler.get_next_command()
                
                if command:
                    # Update status
                    command.status = TaskStatus.EXECUTING
                    self.task_scheduler.executing_tasks[command.command_id] = command
                    
                    # Publish command
                    message = WeldersArcMessage(
                        message_id=command.command_id,
                        timestamp=datetime.utcnow(),
                        subsystem=SubsystemID.SS3_COMMAND,
                        event_type="command_execution",
                        data={
                            "command_id": command.command_id,
                            "command_type": command.command_type.value,
                            "target_asset": command.target_asset,
                            "parameters": command.parameters,
                            "priority": command.priority.value
                        }
                    )
                    
                    await self.kafka_client.publish(KafkaTopics.COMMAND_EXECUTION, message)
                    
                    # Track in history
                    self.command_history.append(command)
                    self.metrics["commands_issued"] += 1
                    
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in command executor: {e}")
                await asyncio.sleep(5)
                
    async def _plan_monitor(self):
        """Monitor active plans"""
        while True:
            try:
                current_time = datetime.utcnow()
                
                for plan_id, plan in list(self.active_plans.items()):
                    # Check plan status
                    completed_commands = sum(1 for cmd in plan.commands 
                                           if cmd.status == TaskStatus.COMPLETED)
                    failed_commands = sum(1 for cmd in plan.commands 
                                        if cmd.status == TaskStatus.FAILED)
                    
                    if completed_commands == len(plan.commands):
                        plan.status = TaskStatus.COMPLETED
                        plan.metrics["completion_time"] = current_time
                        plan.metrics["success_rate"] = 1.0
                        
                        # Remove from active plans
                        del self.active_plans[plan_id]
                        
                    elif failed_commands > len(plan.commands) * 0.5:
                        plan.status = TaskStatus.FAILED
                        plan.metrics["failure_time"] = current_time
                        plan.metrics["success_rate"] = completed_commands / len(plan.commands)
                        
                        # Remove from active plans
                        del self.active_plans[plan_id]
                        
                    elif current_time > plan.end_time:
                        plan.status = TaskStatus.CANCELLED
                        del self.active_plans[plan_id]
                        
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in plan monitor: {e}")
                await asyncio.sleep(30)
                
    async def _metrics_reporter(self):
        """Report C2 metrics"""
        while True:
            try:
                # Calculate metrics
                total_commands = self.metrics["commands_issued"]
                if total_commands > 0:
                    success_rate = self.metrics["commands_completed"] / total_commands
                    failure_rate = self.metrics["commands_failed"] / total_commands
                else:
                    success_rate = 0
                    failure_rate = 0
                    
                # Asset utilization
                total_assets = len(self.assets)
                busy_assets = sum(1 for a in self.assets.values() if a.current_task)
                utilization = busy_assets / total_assets if total_assets > 0 else 0
                
                # Publish metrics
                message = WeldersArcMessage(
                    message_id=f"c2-metrics-{datetime.utcnow().timestamp()}",
                    timestamp=datetime.utcnow(),
                    subsystem=SubsystemID.SS3_COMMAND,
                    event_type="metrics_report",
                    data={
                        "commands_issued": total_commands,
                        "success_rate": success_rate,
                        "failure_rate": failure_rate,
                        "active_plans": len(self.active_plans),
                        "asset_utilization": utilization,
                        "defensive_posture": self.battle_manager.current_posture
                    }
                )
                
                await self.kafka_client.publish(KafkaTopics.METRICS, message)
                
                await asyncio.sleep(60)  # Report every minute
                
            except Exception as e:
                logger.error(f"Error in metrics reporter: {e}")
                await asyncio.sleep(60)
                
    async def execute_plan(self, plan: TaskPlan):
        """Execute a task plan"""
        self.active_plans[plan.plan_id] = plan
        
        # Schedule all commands
        for command in plan.commands:
            success = await self.task_scheduler.schedule_command(command, self.assets)
            if not success:
                logger.warning(f"Failed to schedule command {command.command_id}")
                
        # Publish plan execution
        message = WeldersArcMessage(
            message_id=plan.plan_id,
            timestamp=datetime.utcnow(),
            subsystem=SubsystemID.SS3_COMMAND,
            event_type="plan_execution",
            data={
                "plan_id": plan.plan_id,
                "objective": plan.objective,
                "command_count": len(plan.commands),
                "priority": plan.priority.value,
                "assets": plan.assets_involved
            }
        )
        
        await self.kafka_client.publish(KafkaTopics.PLAN_EXECUTION, message)
        
    async def change_defensive_posture(self, new_posture: str):
        """Change defensive posture"""
        if new_posture not in self.battle_manager.defensive_postures:
            logger.error(f"Unknown posture: {new_posture}")
            return
            
        self.battle_manager.current_posture = new_posture
        
        # Generate posture commands
        posture_func = self.battle_manager.defensive_postures[new_posture]
        commands = await posture_func(self.assets)
        
        # Create posture plan
        plan = TaskPlan(
            plan_id=f"posture-{new_posture}-{datetime.utcnow().timestamp()}",
            objective=f"Transition to {new_posture} defensive posture",
            commands=commands,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(hours=8),
            assets_involved=[cmd.target_asset for cmd in commands],
            priority=CommandPriority.HIGH
        )
        
        await self.execute_plan(plan)
        
    def get_asset_status(self) -> Dict[str, Any]:
        """Get current asset status"""
        return {
            "total_assets": len(self.assets),
            "asset_types": {
                asset_type.value: sum(1 for a in self.assets.values() 
                                     if a.asset_type == asset_type)
                for asset_type in AssetType
            },
            "busy_assets": sum(1 for a in self.assets.values() if a.current_task),
            "available_assets": sum(1 for a in self.assets.values() 
                                   if not a.current_task and a.availability > 0.5)
        }
        
    def get_command_queue_status(self) -> Dict[str, Any]:
        """Get command queue status"""
        queue_by_priority = defaultdict(int)
        for cmd in self.task_scheduler.schedule_queue:
            queue_by_priority[cmd.priority.value] += 1
            
        return {
            "queued_commands": len(self.task_scheduler.schedule_queue),
            "executing_commands": len(self.task_scheduler.executing_tasks),
            "queue_by_priority": dict(queue_by_priority)
        } 