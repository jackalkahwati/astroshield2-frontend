"""
Subsystem 6: Threat Assessment & Response Coordination
Coordinate defensive and offensive responses to space threats
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from enum import Enum
import numpy as np
from collections import defaultdict, deque

from ..kafka.kafka_client import (
    WeldersArcKafkaClient,
    WeldersArcMessage,
    KafkaTopics,
    SubsystemID,
    EventType
)
from .ss1_target_modeling import TargetModel, BehaviorPattern, ObjectType
from .ss3_command_control import Command, CommandType, CommandPriority, TaskStatus
from .ss5_hostility_monitoring import IntentLevel, ThreatCategory

logger = logging.getLogger(__name__)


class ResponseType(Enum):
    """Types of responses"""
    MONITOR = "monitor"
    TRACK = "track"
    COLLECT = "collect"
    DEFENSIVE = "defensive"
    EVASIVE = "evasive"
    JAMMING = "jamming"
    KINETIC = "kinetic"
    CYBER = "cyber"
    DIPLOMATIC = "diplomatic"


class ThreatLevel(Enum):
    """Assessed threat levels"""
    MINIMAL = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    CRITICAL = 5


class ResponseStatus(Enum):
    """Response execution status"""
    PLANNING = "planning"
    APPROVED = "approved"
    EXECUTING = "executing"
    MONITORING = "monitoring"
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass
class ThreatAssessment:
    """Comprehensive threat assessment"""
    threat_id: str
    object_id: str
    assessment_time: datetime
    threat_level: ThreatLevel
    threat_category: ThreatCategory
    intent_level: IntentLevel
    
    # Threat characteristics
    capabilities: List[str]
    vulnerabilities: List[str]
    likely_objectives: List[str]
    time_to_impact: Optional[timedelta] = None
    
    # Assessment scores
    credibility: float  # 0-1
    severity: float  # 0-1
    imminence: float  # 0-1
    
    # Affected assets
    threatened_assets: List[str] = field(default_factory=list)
    impact_assessment: Dict[str, float] = field(default_factory=dict)
    
    # Intelligence
    data_sources: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ResponseOption:
    """Potential response option"""
    option_id: str
    response_type: ResponseType
    description: str
    effectiveness: float  # 0-1
    risk_level: float  # 0-1
    resource_requirements: Dict[str, int]
    estimated_duration: timedelta
    reversibility: bool
    escalation_risk: float  # 0-1
    commands: List[Command] = field(default_factory=list)


@dataclass
class ResponsePlan:
    """Coordinated response plan"""
    plan_id: str
    threat_id: str
    created_time: datetime
    approved_time: Optional[datetime] = None
    
    # Plan details
    primary_response: ResponseOption
    backup_responses: List[ResponseOption] = field(default_factory=list)
    escalation_ladder: List[ResponseOption] = field(default_factory=list)
    
    # Coordination
    participating_subsystems: List[SubsystemID] = field(default_factory=list)
    required_assets: List[str] = field(default_factory=list)
    coordination_requirements: List[str] = field(default_factory=list)
    
    # Execution
    status: ResponseStatus = ResponseStatus.PLANNING
    execution_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Decision criteria
    success_criteria: List[str] = field(default_factory=list)
    abort_criteria: List[str] = field(default_factory=list)
    
    # Authorization
    authorization_level: str = "tactical"
    approvers: List[str] = field(default_factory=list)


class ThreatAnalyzer:
    """Analyze and assess threats"""
    
    def __init__(self):
        self.assessment_history: deque = deque(maxlen=1000)
        self.threat_patterns: Dict[str, List[ThreatAssessment]] = defaultdict(list)
        
    async def assess_threat(
        self,
        target: TargetModel,
        intent_assessment: Dict[str, Any],
        wez_prediction: Dict[str, Any]
    ) -> ThreatAssessment:
        """Perform comprehensive threat assessment"""
        
        # Determine threat level
        threat_level = self._calculate_threat_level(
            target.anomaly_score,
            intent_assessment.get("intent_level", IntentLevel.UNKNOWN),
            wez_prediction.get("threat_category", ThreatCategory.LOW)
        )
        
        # Assess capabilities
        capabilities = await self._assess_capabilities(target)
        
        # Identify vulnerabilities
        vulnerabilities = await self._identify_vulnerabilities(target)
        
        # Determine likely objectives
        objectives = await self._determine_objectives(target, intent_assessment)
        
        # Calculate time to impact
        time_to_impact = None
        if wez_prediction.get("time_to_wez"):
            time_to_impact = timedelta(seconds=wez_prediction["time_to_wez"])
            
        # Score the threat
        credibility = self._calculate_credibility(target, intent_assessment)
        severity = self._calculate_severity(threat_level, capabilities)
        imminence = self._calculate_imminence(time_to_impact, target.behavior_pattern)
        
        # Identify threatened assets
        threatened_assets = await self._identify_threatened_assets(
            target,
            wez_prediction.get("affected_objects", [])
        )
        
        # Create assessment
        assessment = ThreatAssessment(
            threat_id=f"threat-{target.object_id}-{datetime.utcnow().timestamp()}",
            object_id=target.object_id,
            assessment_time=datetime.utcnow(),
            threat_level=threat_level,
            threat_category=ThreatCategory(intent_assessment.get("threat_category", "low")),
            intent_level=IntentLevel(intent_assessment.get("intent_level", "unknown")),
            capabilities=capabilities,
            vulnerabilities=vulnerabilities,
            likely_objectives=objectives,
            time_to_impact=time_to_impact,
            credibility=credibility,
            severity=severity,
            imminence=imminence,
            threatened_assets=threatened_assets,
            confidence=target.pattern_confidence * credibility
        )
        
        # Store in history
        self.assessment_history.append(assessment)
        self.threat_patterns[target.behavior_pattern].append(assessment)
        
        return assessment
        
    def _calculate_threat_level(
        self,
        anomaly_score: float,
        intent_level: IntentLevel,
        threat_category: ThreatCategory
    ) -> ThreatLevel:
        """Calculate overall threat level"""
        # Base score from anomaly
        score = anomaly_score * 2  # 0-2
        
        # Add intent level
        intent_scores = {
            IntentLevel.BENIGN: 0,
            IntentLevel.UNKNOWN: 1,
            IntentLevel.SUSPICIOUS: 2,
            IntentLevel.CONCERNING: 3,
            IntentLevel.HOSTILE: 4,
            IntentLevel.CRITICAL: 5
        }
        score += intent_scores.get(intent_level, 1) * 0.5  # 0-2.5
        
        # Add threat category
        category_scores = {
            ThreatCategory.NONE: 0,
            ThreatCategory.LOW: 0.5,
            ThreatCategory.MEDIUM: 1,
            ThreatCategory.HIGH: 1.5,
            ThreatCategory.CRITICAL: 2
        }
        score += category_scores.get(threat_category, 0.5)  # 0-2
        
        # Total score 0-6.5, map to threat levels
        if score < 1:
            return ThreatLevel.MINIMAL
        elif score < 2:
            return ThreatLevel.LOW
        elif score < 3.5:
            return ThreatLevel.MODERATE
        elif score < 5:
            return ThreatLevel.HIGH
        else:
            return ThreatLevel.CRITICAL
            
    async def _assess_capabilities(self, target: TargetModel) -> List[str]:
        """Assess threat capabilities"""
        capabilities = []
        
        # Maneuver capability
        if target.maneuver_capability and target.maneuver_capability.delta_v_budget > 50:
            capabilities.append("high_maneuverability")
            
        # Object type capabilities
        if target.object_type == ObjectType.WEAPON:
            capabilities.extend(["kinetic_strike", "explosive"])
        elif target.object_type == ObjectType.ACTIVE_SATELLITE:
            capabilities.extend(["sensors", "communications"])
            
        # Behavior-based capabilities
        if target.behavior_pattern == BehaviorPattern.AGGRESSIVE:
            capabilities.append("offensive_posture")
        elif target.behavior_pattern == BehaviorPattern.EVASIVE:
            capabilities.append("stealth_operations")
            
        # Threat indicators
        if "high_power_rf" in target.threat_indicators:
            capabilities.append("rf_weapons")
        if "cyber_activity" in target.threat_indicators:
            capabilities.append("cyber_warfare")
            
        return capabilities
        
    async def _identify_vulnerabilities(self, target: TargetModel) -> List[str]:
        """Identify target vulnerabilities"""
        vulnerabilities = []
        
        # Fuel limitations
        if target.maneuver_capability and target.maneuver_capability.fuel_remaining < 10:
            vulnerabilities.append("low_fuel")
            
        # Predictable behavior
        if target.behavior_pattern == BehaviorPattern.STATION_KEEPING:
            vulnerabilities.append("predictable_orbit")
            
        # Communication requirements
        if target.object_type == ObjectType.ACTIVE_SATELLITE:
            vulnerabilities.append("command_link_dependency")
            
        # Physical vulnerabilities
        if target.cross_sectional_area > 50:
            vulnerabilities.append("large_target")
            
        return vulnerabilities
        
    async def _determine_objectives(
        self,
        target: TargetModel,
        intent_assessment: Dict[str, Any]
    ) -> List[str]:
        """Determine likely threat objectives"""
        objectives = []
        
        # Based on behavior
        behavior_objectives = {
            BehaviorPattern.RENDEZVOUS: ["inspection", "capture", "interference"],
            BehaviorPattern.PROXIMITY_OPS: ["surveillance", "jamming", "docking"],
            BehaviorPattern.AGGRESSIVE: ["attack", "disable", "destroy"],
            BehaviorPattern.EVASIVE: ["infiltration", "covert_ops", "deception"]
        }
        
        if target.behavior_pattern in behavior_objectives:
            objectives.extend(behavior_objectives[target.behavior_pattern])
            
        # Based on intent
        if intent_assessment.get("intent_level") == IntentLevel.HOSTILE:
            objectives.extend(["disruption", "denial_of_service"])
            
        return objectives
        
    def _calculate_credibility(
        self,
        target: TargetModel,
        intent_assessment: Dict[str, Any]
    ) -> float:
        """Calculate threat credibility"""
        credibility = 0.5  # Base credibility
        
        # Pattern confidence
        credibility += target.pattern_confidence * 0.3
        
        # Multiple data sources
        if len(target.data_sources) > 3:
            credibility += 0.2
            
        # Consistent behavior
        if target.behavior_pattern != BehaviorPattern.ANOMALOUS:
            credibility += 0.1
            
        return min(1.0, credibility)
        
    def _calculate_severity(self, threat_level: ThreatLevel, capabilities: List[str]) -> float:
        """Calculate potential severity"""
        # Base severity from threat level
        severity_map = {
            ThreatLevel.MINIMAL: 0.1,
            ThreatLevel.LOW: 0.3,
            ThreatLevel.MODERATE: 0.5,
            ThreatLevel.HIGH: 0.7,
            ThreatLevel.CRITICAL: 0.9
        }
        severity = severity_map[threat_level]
        
        # Adjust for dangerous capabilities
        dangerous_capabilities = ["kinetic_strike", "explosive", "rf_weapons", "cyber_warfare"]
        for cap in capabilities:
            if cap in dangerous_capabilities:
                severity = min(1.0, severity + 0.1)
                
        return severity
        
    def _calculate_imminence(
        self,
        time_to_impact: Optional[timedelta],
        behavior: BehaviorPattern
    ) -> float:
        """Calculate threat imminence"""
        if not time_to_impact:
            # Use behavior as proxy
            if behavior in [BehaviorPattern.AGGRESSIVE, BehaviorPattern.PROXIMITY_OPS]:
                return 0.7
            else:
                return 0.3
                
        # Time-based imminence
        hours_to_impact = time_to_impact.total_seconds() / 3600
        
        if hours_to_impact < 1:
            return 1.0
        elif hours_to_impact < 6:
            return 0.8
        elif hours_to_impact < 24:
            return 0.6
        elif hours_to_impact < 72:
            return 0.4
        else:
            return 0.2
            
    async def _identify_threatened_assets(
        self,
        target: TargetModel,
        affected_objects: List[str]
    ) -> List[str]:
        """Identify assets threatened by target"""
        threatened = list(affected_objects)
        
        # Add high-value assets in same orbital regime
        if target.semi_major_axis < 8000:  # LEO
            threatened.extend(["leo_constellation", "imaging_satellites"])
        elif 35786 - 100 < target.semi_major_axis < 35786 + 100:  # GEO
            threatened.extend(["geo_comms", "missile_warning"])
            
        return threatened


class ResponsePlanner:
    """Plan and coordinate responses"""
    
    def __init__(self):
        self.response_library = self._build_response_library()
        self.active_responses: Dict[str, ResponsePlan] = {}
        
    def _build_response_library(self) -> Dict[ResponseType, List[ResponseOption]]:
        """Build library of response options"""
        return {
            ResponseType.MONITOR: [
                ResponseOption(
                    option_id="monitor_basic",
                    response_type=ResponseType.MONITOR,
                    description="Basic monitoring and tracking",
                    effectiveness=0.3,
                    risk_level=0.0,
                    resource_requirements={"sensors": 1},
                    estimated_duration=timedelta(hours=24),
                    reversibility=True,
                    escalation_risk=0.0
                ),
                ResponseOption(
                    option_id="monitor_enhanced",
                    response_type=ResponseType.MONITOR,
                    description="Enhanced multi-sensor monitoring",
                    effectiveness=0.6,
                    risk_level=0.0,
                    resource_requirements={"sensors": 3},
                    estimated_duration=timedelta(hours=12),
                    reversibility=True,
                    escalation_risk=0.0
                )
            ],
            ResponseType.DEFENSIVE: [
                ResponseOption(
                    option_id="defensive_maneuver",
                    response_type=ResponseType.DEFENSIVE,
                    description="Defensive orbital maneuver",
                    effectiveness=0.7,
                    risk_level=0.2,
                    resource_requirements={"satellites": 1, "fuel": 10},
                    estimated_duration=timedelta(hours=2),
                    reversibility=True,
                    escalation_risk=0.1
                ),
                ResponseOption(
                    option_id="defensive_formation",
                    response_type=ResponseType.DEFENSIVE,
                    description="Defensive formation flying",
                    effectiveness=0.8,
                    risk_level=0.3,
                    resource_requirements={"satellites": 3, "fuel": 30},
                    estimated_duration=timedelta(hours=6),
                    reversibility=True,
                    escalation_risk=0.2
                )
            ],
            ResponseType.JAMMING: [
                ResponseOption(
                    option_id="rf_jamming",
                    response_type=ResponseType.JAMMING,
                    description="RF command link jamming",
                    effectiveness=0.6,
                    risk_level=0.4,
                    resource_requirements={"rf_systems": 1},
                    estimated_duration=timedelta(hours=1),
                    reversibility=True,
                    escalation_risk=0.5
                )
            ],
            ResponseType.KINETIC: [
                ResponseOption(
                    option_id="kinetic_intercept",
                    response_type=ResponseType.KINETIC,
                    description="Kinetic intercept",
                    effectiveness=0.9,
                    risk_level=0.8,
                    resource_requirements={"interceptors": 1},
                    estimated_duration=timedelta(minutes=30),
                    reversibility=False,
                    escalation_risk=0.9
                )
            ]
        }
        
    async def generate_response_plan(
        self,
        assessment: ThreatAssessment,
        constraints: Dict[str, Any]
    ) -> ResponsePlan:
        """Generate response plan for threat"""
        
        # Select appropriate response types based on threat level
        response_types = self._select_response_types(assessment)
        
        # Generate response options
        options = []
        for response_type in response_types:
            type_options = self.response_library.get(response_type, [])
            for option in type_options:
                if await self._validate_option(option, assessment, constraints):
                    options.append(option)
                    
        # Rank options
        ranked_options = self._rank_options(options, assessment)
        
        if not ranked_options:
            # Default to monitoring
            ranked_options = [self.response_library[ResponseType.MONITOR][0]]
            
        # Build escalation ladder
        escalation_ladder = self._build_escalation_ladder(ranked_options)
        
        # Create plan
        plan = ResponsePlan(
            plan_id=f"response-{assessment.threat_id}-{datetime.utcnow().timestamp()}",
            threat_id=assessment.threat_id,
            created_time=datetime.utcnow(),
            primary_response=ranked_options[0],
            backup_responses=ranked_options[1:3] if len(ranked_options) > 1 else [],
            escalation_ladder=escalation_ladder,
            participating_subsystems=self._identify_required_subsystems(ranked_options[0]),
            required_assets=self._identify_required_assets(ranked_options[0]),
            success_criteria=self._define_success_criteria(assessment, ranked_options[0]),
            abort_criteria=self._define_abort_criteria(assessment, ranked_options[0]),
            authorization_level=self._determine_authorization_level(ranked_options[0])
        )
        
        return plan
        
    def _select_response_types(self, assessment: ThreatAssessment) -> List[ResponseType]:
        """Select appropriate response types"""
        response_types = [ResponseType.MONITOR]  # Always monitor
        
        if assessment.threat_level == ThreatLevel.MINIMAL:
            pass  # Just monitor
        elif assessment.threat_level == ThreatLevel.LOW:
            response_types.append(ResponseType.TRACK)
        elif assessment.threat_level == ThreatLevel.MODERATE:
            response_types.extend([ResponseType.TRACK, ResponseType.COLLECT])
        elif assessment.threat_level == ThreatLevel.HIGH:
            response_types.extend([
                ResponseType.TRACK,
                ResponseType.COLLECT,
                ResponseType.DEFENSIVE,
                ResponseType.JAMMING
            ])
        else:  # CRITICAL
            response_types.extend([
                ResponseType.TRACK,
                ResponseType.DEFENSIVE,
                ResponseType.JAMMING,
                ResponseType.KINETIC
            ])
            
        return response_types
        
    async def _validate_option(
        self,
        option: ResponseOption,
        assessment: ThreatAssessment,
        constraints: Dict[str, Any]
    ) -> bool:
        """Validate if response option is feasible"""
        # Check resource availability
        for resource, required in option.resource_requirements.items():
            available = constraints.get(f"available_{resource}", 0)
            if available < required:
                return False
                
        # Check authorization constraints
        if option.response_type == ResponseType.KINETIC:
            if not constraints.get("kinetic_authorized", False):
                return False
                
        # Check escalation constraints
        max_escalation = constraints.get("max_escalation_risk", 1.0)
        if option.escalation_risk > max_escalation:
            return False
            
        return True
        
    def _rank_options(
        self,
        options: List[ResponseOption],
        assessment: ThreatAssessment
    ) -> List[ResponseOption]:
        """Rank response options"""
        scored_options = []
        
        for option in options:
            # Score based on effectiveness vs risk
            score = option.effectiveness * (1 - option.risk_level)
            
            # Adjust for threat characteristics
            if assessment.imminence > 0.8:
                # Prefer faster responses
                if option.estimated_duration < timedelta(hours=1):
                    score *= 1.2
                    
            if assessment.severity > 0.8:
                # Accept higher risk for severe threats
                score = option.effectiveness * (1 - option.risk_level * 0.5)
                
            scored_options.append((score, option))
            
        # Sort by score
        scored_options.sort(key=lambda x: x[0], reverse=True)
        
        return [option for score, option in scored_options]
        
    def _build_escalation_ladder(self, options: List[ResponseOption]) -> List[ResponseOption]:
        """Build escalation ladder from options"""
        # Sort by escalation risk
        escalation_sorted = sorted(options, key=lambda x: x.escalation_risk)
        
        # Select diverse options for ladder
        ladder = []
        last_risk = -1
        for option in escalation_sorted:
            if option.escalation_risk > last_risk + 0.2:
                ladder.append(option)
                last_risk = option.escalation_risk
                
        return ladder
        
    def _identify_required_subsystems(self, option: ResponseOption) -> List[SubsystemID]:
        """Identify subsystems required for response"""
        subsystems = [SubsystemID.SS6_RESPONSE]  # Always involved
        
        if option.response_type in [ResponseType.MONITOR, ResponseType.TRACK]:
            subsystems.extend([SubsystemID.SS0_INGESTION, SubsystemID.SS2_ESTIMATION])
        elif option.response_type == ResponseType.DEFENSIVE:
            subsystems.extend([SubsystemID.SS3_COMMAND, SubsystemID.SS4_CCDM])
        elif option.response_type == ResponseType.JAMMING:
            subsystems.extend([SubsystemID.SS3_COMMAND, SubsystemID.SS5_HOSTILITY])
            
        return list(set(subsystems))
        
    def _identify_required_assets(self, option: ResponseOption) -> List[str]:
        """Identify assets required for response"""
        assets = []
        
        for resource, count in option.resource_requirements.items():
            for i in range(count):
                assets.append(f"{resource}_{i}")
                
        return assets
        
    def _define_success_criteria(
        self,
        assessment: ThreatAssessment,
        option: ResponseOption
    ) -> List[str]:
        """Define success criteria for response"""
        criteria = []
        
        if option.response_type == ResponseType.MONITOR:
            criteria.append("maintain_track_quality > 0.8")
        elif option.response_type == ResponseType.DEFENSIVE:
            criteria.append("increase_miss_distance > 100km")
            criteria.append("no_collision_predicted")
        elif option.response_type == ResponseType.JAMMING:
            criteria.append("command_link_disrupted")
            criteria.append("threat_behavior_changed")
            
        # General criteria
        criteria.append(f"threat_level_reduced_below_{assessment.threat_level.value - 1}")
        
        return criteria
        
    def _define_abort_criteria(
        self,
        assessment: ThreatAssessment,
        option: ResponseOption
    ) -> List[str]:
        """Define abort criteria for response"""
        criteria = []
        
        # General abort criteria
        criteria.extend([
            "friendly_asset_endangered",
            "collateral_damage_predicted",
            "diplomatic_intervention"
        ])
        
        # Option-specific criteria
        if option.response_type == ResponseType.KINETIC:
            criteria.append("debris_field_unacceptable")
            criteria.append("target_identified_as_friendly")
            
        return criteria
        
    def _determine_authorization_level(self, option: ResponseOption) -> str:
        """Determine required authorization level"""
        if option.response_type in [ResponseType.MONITOR, ResponseType.TRACK]:
            return "tactical"
        elif option.response_type in [ResponseType.DEFENSIVE, ResponseType.COLLECT]:
            return "operational"
        elif option.response_type == ResponseType.JAMMING:
            return "strategic"
        elif option.response_type == ResponseType.KINETIC:
            return "national"
        else:
            return "tactical"


class ResponseCoordinator:
    """Coordinate response execution"""
    
    def __init__(self):
        self.active_responses: Dict[str, ResponsePlan] = {}
        self.response_metrics: Dict[str, Dict[str, Any]] = {}
        self.coordination_state: Dict[str, Any] = {}
        
    async def execute_response(
        self,
        plan: ResponsePlan,
        subsystems: Dict[SubsystemID, Any]
    ) -> bool:
        """Execute response plan"""
        plan.status = ResponseStatus.EXECUTING
        self.active_responses[plan.plan_id] = plan
        
        try:
            # Initialize coordination
            await self._initialize_coordination(plan, subsystems)
            
            # Execute primary response
            success = await self._execute_option(plan.primary_response, subsystems)
            
            if not success and plan.backup_responses:
                # Try backup options
                for backup in plan.backup_responses:
                    success = await self._execute_option(backup, subsystems)
                    if success:
                        break
                        
            # Update metrics
            plan.execution_metrics["success"] = success
            plan.execution_metrics["end_time"] = datetime.utcnow()
            
            # Monitor results
            if success:
                plan.status = ResponseStatus.MONITORING
                asyncio.create_task(self._monitor_response(plan))
            else:
                plan.status = ResponseStatus.ABORTED
                
            return success
            
        except Exception as e:
            logger.error(f"Error executing response: {e}")
            plan.status = ResponseStatus.ABORTED
            return False
            
    async def _initialize_coordination(
        self,
        plan: ResponsePlan,
        subsystems: Dict[SubsystemID, Any]
    ):
        """Initialize subsystem coordination"""
        # Create coordination state
        self.coordination_state[plan.plan_id] = {
            "subsystems_ready": {},
            "resources_allocated": {},
            "synchronization_points": {},
            "abort_requested": False
        }
        
        # Notify participating subsystems
        for subsystem_id in plan.participating_subsystems:
            if subsystem_id in subsystems:
                # Send coordination request
                ready = await self._request_subsystem_participation(
                    subsystem_id,
                    plan,
                    subsystems[subsystem_id]
                )
                self.coordination_state[plan.plan_id]["subsystems_ready"][subsystem_id] = ready
                
    async def _execute_option(
        self,
        option: ResponseOption,
        subsystems: Dict[SubsystemID, Any]
    ) -> bool:
        """Execute a response option"""
        # Generate commands for option
        commands = await self._generate_commands(option)
        option.commands = commands
        
        # Send commands to C2
        if SubsystemID.SS3_COMMAND in subsystems:
            c2_system = subsystems[SubsystemID.SS3_COMMAND]
            
            for command in commands:
                await c2_system.task_scheduler.schedule_command(
                    command,
                    c2_system.assets
                )
                
            # Wait for execution
            success = await self._wait_for_execution(commands, option.estimated_duration)
            return success
            
        return False
        
    async def _generate_commands(self, option: ResponseOption) -> List[Command]:
        """Generate commands for response option"""
        commands = []
        
        if option.response_type == ResponseType.MONITOR:
            # Generate monitoring commands
            for i, sensor in enumerate(option.resource_requirements.get("sensors", [])):
                commands.append(Command(
                    command_id=f"{option.option_id}-sensor-{i}-{datetime.utcnow().timestamp()}",
                    command_type=CommandType.SENSOR_TASKING,
                    priority=CommandPriority.HIGH,
                    target_asset=f"sensor_{i}",
                    parameters={
                        "mode": "track",
                        "priority": "high",
                        "duration": option.estimated_duration.total_seconds()
                    },
                    issuer="response_coordinator",
                    issued_time=datetime.utcnow(),
                    execution_time=datetime.utcnow(),
                    expiry_time=datetime.utcnow() + option.estimated_duration
                ))
                
        elif option.response_type == ResponseType.DEFENSIVE:
            # Generate defensive maneuver commands
            commands.append(Command(
                command_id=f"{option.option_id}-maneuver-{datetime.utcnow().timestamp()}",
                command_type=CommandType.MANEUVER_TASK,
                priority=CommandPriority.CRITICAL,
                target_asset="satellite_1",
                parameters={
                    "maneuver_type": "defensive",
                    "delta_v": option.resource_requirements.get("fuel", 10)
                },
                issuer="response_coordinator",
                issued_time=datetime.utcnow(),
                execution_time=datetime.utcnow() + timedelta(minutes=5),
                expiry_time=datetime.utcnow() + timedelta(hours=1)
            ))
            
        return commands
        
    async def _wait_for_execution(
        self,
        commands: List[Command],
        timeout: timedelta
    ) -> bool:
        """Wait for command execution completion"""
        start_time = datetime.utcnow()
        
        while datetime.utcnow() - start_time < timeout:
            # Check command status
            completed = sum(1 for cmd in commands if cmd.status == TaskStatus.COMPLETED)
            failed = sum(1 for cmd in commands if cmd.status == TaskStatus.FAILED)
            
            if completed == len(commands):
                return True
            elif failed > len(commands) * 0.5:
                return False
                
            await asyncio.sleep(5)
            
        # Timeout
        return False
        
    async def _monitor_response(self, plan: ResponsePlan):
        """Monitor response effectiveness"""
        monitoring_duration = timedelta(hours=1)
        start_time = datetime.utcnow()
        
        while datetime.utcnow() - start_time < monitoring_duration:
            # Check success criteria
            criteria_met = await self._check_success_criteria(plan)
            
            if all(criteria_met.values()):
                plan.status = ResponseStatus.COMPLETED
                plan.execution_metrics["success_criteria_met"] = True
                break
                
            # Check abort criteria
            abort_triggered = await self._check_abort_criteria(plan)
            
            if any(abort_triggered.values()):
                plan.status = ResponseStatus.ABORTED
                plan.execution_metrics["abort_reason"] = abort_triggered
                await self._abort_response(plan)
                break
                
            await asyncio.sleep(30)
            
    async def _check_success_criteria(self, plan: ResponsePlan) -> Dict[str, bool]:
        """Check if success criteria are met"""
        results = {}
        
        for criterion in plan.success_criteria:
            # Simplified check - in production would query actual metrics
            results[criterion] = np.random.random() > 0.3
            
        return results
        
    async def _check_abort_criteria(self, plan: ResponsePlan) -> Dict[str, bool]:
        """Check if abort criteria are triggered"""
        results = {}
        
        for criterion in plan.abort_criteria:
            # Simplified check
            results[criterion] = np.random.random() > 0.95
            
        return results
        
    async def _abort_response(self, plan: ResponsePlan):
        """Abort response execution"""
        # Cancel remaining commands
        for option in [plan.primary_response] + plan.backup_responses:
            for command in option.commands:
                if command.status in [TaskStatus.PENDING, TaskStatus.EXECUTING]:
                    command.status = TaskStatus.CANCELLED
                    
        # Notify subsystems
        self.coordination_state[plan.plan_id]["abort_requested"] = True
        
    async def _request_subsystem_participation(
        self,
        subsystem_id: SubsystemID,
        plan: ResponsePlan,
        subsystem: Any
    ) -> bool:
        """Request subsystem participation in response"""
        # Simplified - in production would have proper subsystem interfaces
        return True


class ThreatResponseSubsystem:
    """Main threat assessment and response coordination subsystem"""
    
    def __init__(self, kafka_client: WeldersArcKafkaClient):
        self.kafka_client = kafka_client
        self.threat_analyzer = ThreatAnalyzer()
        self.response_planner = ResponsePlanner()
        self.response_coordinator = ResponseCoordinator()
        self.active_threats: Dict[str, ThreatAssessment] = {}
        self.subsystem_references: Dict[SubsystemID, Any] = {}
        
    async def initialize(self, subsystems: Dict[SubsystemID, Any] = None):
        """Initialize threat response subsystem"""
        if subsystems:
            self.subsystem_references = subsystems
            
        # Subscribe to relevant topics
        self.kafka_client.subscribe(
            KafkaTopics.ALERT_THREAT,
            self._handle_threat_alert
        )
        
        self.kafka_client.subscribe(
            KafkaTopics.HOSTILE_INTENT,
            self._handle_intent_update
        )
        
        self.kafka_client.subscribe(
            KafkaTopics.WEZ_PREDICTION,
            self._handle_wez_update
        )
        
        # Start background tasks
        asyncio.create_task(self._threat_monitor())
        asyncio.create_task(self._response_monitor())
        
        logger.info("Threat response subsystem initialized")
        
    async def _handle_threat_alert(self, message: WeldersArcMessage):
        """Handle incoming threat alert"""
        try:
            # Create simplified target model
            target = TargetModel(
                object_id=message.data["object_id"],
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
                behavior_pattern=BehaviorPattern(message.data.get("behavior", "anomalous")),
                pattern_confidence=message.data.get("confidence", 0.5),
                maneuver_capability=None,
                anomaly_score=message.data.get("threat_level", 0.5)
            )
            
            # Get related assessments
            intent_assessment = message.data.get("intent_assessment", {})
            wez_prediction = message.data.get("wez_prediction", {})
            
            # Perform threat assessment
            assessment = await self.threat_analyzer.assess_threat(
                target,
                intent_assessment,
                wez_prediction
            )
            
            self.active_threats[assessment.threat_id] = assessment
            
            # Generate response plan if threat is significant
            if assessment.threat_level.value >= ThreatLevel.MODERATE.value:
                await self._plan_response(assessment)
                
        except Exception as e:
            logger.error(f"Error handling threat alert: {e}")
            
    async def _handle_intent_update(self, message: WeldersArcMessage):
        """Handle intent assessment update"""
        object_id = message.data["object_id"]
        
        # Update relevant threat assessments
        for threat_id, assessment in self.active_threats.items():
            if assessment.object_id == object_id:
                assessment.intent_level = IntentLevel(message.data["intent_level"])
                assessment.threat_category = ThreatCategory(message.data["threat_category"])
                
                # Re-evaluate response if needed
                if assessment.intent_level.value >= IntentLevel.HOSTILE.value:
                    await self._escalate_response(assessment)
                    
    async def _handle_wez_update(self, message: WeldersArcMessage):
        """Handle WEZ prediction update"""
        object_id = message.data["threat_id"]
        
        # Update time to impact for relevant assessments
        for threat_id, assessment in self.active_threats.items():
            if assessment.object_id == object_id:
                if message.data.get("time_to_wez"):
                    assessment.time_to_impact = timedelta(seconds=message.data["time_to_wez"])
                    assessment.imminence = self.threat_analyzer._calculate_imminence(
                        assessment.time_to_impact,
                        BehaviorPattern.AGGRESSIVE
                    )
                    
    async def _plan_response(self, assessment: ThreatAssessment):
        """Plan response to threat"""
        # Define constraints
        constraints = {
            "available_sensors": 5,
            "available_satellites": 3,
            "available_interceptors": 2,
            "available_rf_systems": 1,
            "kinetic_authorized": assessment.threat_level == ThreatLevel.CRITICAL,
            "max_escalation_risk": 0.7 if assessment.threat_level.value < 4 else 1.0
        }
        
        # Generate response plan
        plan = await self.response_planner.generate_response_plan(assessment, constraints)
        
        # Request authorization if needed
        if plan.authorization_level in ["strategic", "national"]:
            await self._request_authorization(plan)
        else:
            # Auto-approve tactical responses
            plan.approved_time = datetime.utcnow()
            plan.status = ResponseStatus.APPROVED
            
            # Execute response
            await self._execute_response(plan)
            
    async def _request_authorization(self, plan: ResponsePlan):
        """Request authorization for response"""
        message = WeldersArcMessage(
            message_id=f"auth-request-{plan.plan_id}",
            timestamp=datetime.utcnow(),
            subsystem=SubsystemID.SS6_RESPONSE,
            event_type="authorization_request",
            data={
                "plan_id": plan.plan_id,
                "threat_id": plan.threat_id,
                "response_type": plan.primary_response.response_type.value,
                "authorization_level": plan.authorization_level,
                "escalation_risk": plan.primary_response.escalation_risk,
                "reversibility": plan.primary_response.reversibility
            }
        )
        
        await self.kafka_client.publish(KafkaTopics.ALERT_OPERATOR, message)
        
    async def _execute_response(self, plan: ResponsePlan):
        """Execute approved response plan"""
        # Store active response
        self.response_coordinator.active_responses[plan.plan_id] = plan
        
        # Execute through coordinator
        success = await self.response_coordinator.execute_response(
            plan,
            self.subsystem_references
        )
        
        # Publish execution status
        message = WeldersArcMessage(
            message_id=f"response-execution-{plan.plan_id}",
            timestamp=datetime.utcnow(),
            subsystem=SubsystemID.SS6_RESPONSE,
            event_type="response_execution",
            data={
                "plan_id": plan.plan_id,
                "threat_id": plan.threat_id,
                "response_type": plan.primary_response.response_type.value,
                "success": success,
                "status": plan.status.value
            }
        )
        
        await self.kafka_client.publish(KafkaTopics.RESPONSE_STATUS, message)
        
    async def _escalate_response(self, assessment: ThreatAssessment):
        """Escalate response for increased threat"""
        # Find active response plan
        active_plan = None
        for plan in self.response_coordinator.active_responses.values():
            if plan.threat_id == assessment.threat_id:
                active_plan = plan
                break
                
        if active_plan and active_plan.escalation_ladder:
            # Move to next escalation level
            current_idx = 0
            for i, option in enumerate(active_plan.escalation_ladder):
                if option == active_plan.primary_response:
                    current_idx = i
                    break
                    
            if current_idx < len(active_plan.escalation_ladder) - 1:
                # Escalate to next level
                next_option = active_plan.escalation_ladder[current_idx + 1]
                
                logger.info(f"Escalating response for {assessment.threat_id} to {next_option.response_type}")
                
                # Execute escalated response
                await self.response_coordinator._execute_option(
                    next_option,
                    self.subsystem_references
                )
                
    async def _threat_monitor(self):
        """Monitor active threats"""
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Check threat status
                for threat_id, assessment in list(self.active_threats.items()):
                    # Remove old threats
                    age = current_time - assessment.assessment_time
                    if age > timedelta(hours=24):
                        del self.active_threats[threat_id]
                        continue
                        
                    # Re-assess if conditions change
                    if age > timedelta(minutes=30):
                        # Would re-query threat data and reassess
                        pass
                        
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in threat monitor: {e}")
                await asyncio.sleep(60)
                
    async def _response_monitor(self):
        """Monitor active responses"""
        while True:
            try:
                # Generate metrics
                active_responses = len(self.response_coordinator.active_responses)
                response_types = defaultdict(int)
                
                for plan in self.response_coordinator.active_responses.values():
                    response_types[plan.primary_response.response_type.value] += 1
                    
                # Publish metrics
                message = WeldersArcMessage(
                    message_id=f"response-metrics-{datetime.utcnow().timestamp()}",
                    timestamp=datetime.utcnow(),
                    subsystem=SubsystemID.SS6_RESPONSE,
                    event_type="metrics_report",
                    data={
                        "active_threats": len(self.active_threats),
                        "active_responses": active_responses,
                        "response_types": dict(response_types),
                        "threat_levels": {
                            level.name: sum(1 for t in self.active_threats.values() 
                                          if t.threat_level == level)
                            for level in ThreatLevel
                        }
                    }
                )
                
                await self.kafka_client.publish(KafkaTopics.METRICS, message)
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in response monitor: {e}")
                await asyncio.sleep(60)
                
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary of current threats"""
        return {
            "total_threats": len(self.active_threats),
            "threat_levels": {
                level.name: sum(1 for t in self.active_threats.values() 
                              if t.threat_level == level)
                for level in ThreatLevel
            },
            "active_responses": len(self.response_coordinator.active_responses),
            "highest_threat": max(
                (t.threat_level.value for t in self.active_threats.values()),
                default=0
            )
        } 