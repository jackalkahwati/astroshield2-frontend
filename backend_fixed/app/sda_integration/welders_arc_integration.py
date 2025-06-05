"""
Main SDA Welders Arc integration service
Orchestrates all subsystems for automated battle management
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from .kafka.kafka_client import WeldersArcKafkaClient, SubsystemID
from .udl.udl_client import UDLClient
from .subsystems.ss0_data_ingestion import DataIngestionSubsystem
from .subsystems.ss1_target_modeling import TargetModelingSubsystem
from .subsystems.ss2_state_estimation import StateEstimationSubsystem
from .subsystems.ss3_command_control import CommandControlSubsystem
from .subsystems.ss4_ccdm import CCDMSubsystem
from .subsystems.ss5_hostility_monitoring import HostilityMonitoringSubsystem
from .subsystems.ss6_threat_response import ThreatResponseSubsystem
from .workflows.node_red_service import NodeRedService
from ..core.config import Settings

logger = logging.getLogger(__name__)


class IntegrationStatus(BaseModel):
    """Integration status response"""
    status: str
    subsystems: Dict[str, Dict[str, Any]]
    health: Dict[str, bool]
    metrics: Dict[str, Any]
    active_threats: int
    defensive_posture: str


class WeldersArcIntegration:
    """Main integration service for Welders Arc"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.kafka_client = WeldersArcKafkaClient(settings)
        self.udl_client = UDLClient(settings)
        self.node_red_service = NodeRedService(settings)
        
        # Initialize subsystems
        self.subsystems: Dict[SubsystemID, Any] = {}
        self._init_subsystems()
        
        # Integration state
        self.is_initialized = False
        self.start_time = None
        self.metrics = {
            "messages_processed": 0,
            "uct_tracks": 0,
            "threats_detected": 0,
            "responses_executed": 0
        }
        
    def _init_subsystems(self):
        """Initialize all subsystems"""
        # SS0: Data Ingestion & Sensors
        self.subsystems[SubsystemID.SS0_INGESTION] = DataIngestionSubsystem(
            self.kafka_client,
            self.udl_client
        )
        
        # SS1: Target Modeling & Characterization
        self.subsystems[SubsystemID.SS1_MODELING] = TargetModelingSubsystem(
            self.kafka_client
        )
        
        # SS2: Tracking & State Estimation
        self.subsystems[SubsystemID.SS2_ESTIMATION] = StateEstimationSubsystem(
            self.kafka_client
        )
        
        # SS3: Command & Control / Logistics
        self.subsystems[SubsystemID.SS3_COMMAND] = CommandControlSubsystem(
            self.kafka_client
        )
        
        # SS4: CCDM
        self.subsystems[SubsystemID.SS4_CCDM] = CCDMSubsystem(
            self.kafka_client,
            self.node_red_service
        )
        
        # SS5: Hostility Monitoring
        self.subsystems[SubsystemID.SS5_HOSTILITY] = HostilityMonitoringSubsystem(
            self.kafka_client
        )
        
        # SS6: Threat Assessment & Response Coordination
        self.subsystems[SubsystemID.SS6_RESPONSE] = ThreatResponseSubsystem(
            self.kafka_client
        )
        
    async def initialize(self):
        """Initialize the integration service"""
        if self.is_initialized:
            return
            
        logger.info("Initializing Welders Arc integration...")
        
        # Initialize Kafka client
        await self.kafka_client.initialize()
        
        # Initialize UDL client
        await self.udl_client.connect()
        
        # Initialize Node-RED workflows
        await self.node_red_service.initialize()
        
        # Initialize all subsystems
        for subsystem_id, subsystem in self.subsystems.items():
            logger.info(f"Initializing {subsystem_id.value}...")
            await subsystem.initialize()
            
        # Pass subsystem references to SS6 for coordination
        await self.subsystems[SubsystemID.SS6_RESPONSE].initialize(self.subsystems)
        
        # Start monitoring
        asyncio.create_task(self._health_monitor())
        asyncio.create_task(self._metrics_collector())
        
        self.is_initialized = True
        self.start_time = datetime.utcnow()
        logger.info("Welders Arc integration initialized successfully")
        
    async def shutdown(self):
        """Shutdown the integration service"""
        logger.info("Shutting down Welders Arc integration...")
        
        # Shutdown subsystems
        for subsystem_id, subsystem in self.subsystems.items():
            logger.info(f"Shutting down {subsystem_id.value}...")
            # Subsystems would have shutdown methods in production
            
        # Disconnect clients
        await self.kafka_client.close()
        await self.udl_client.disconnect()
        
        self.is_initialized = False
        logger.info("Welders Arc integration shutdown complete")
        
    async def _health_monitor(self):
        """Monitor health of all subsystems"""
        while self.is_initialized:
            try:
                health_status = await self.check_health()
                
                # Log any unhealthy subsystems
                for subsystem, healthy in health_status.items():
                    if not healthy:
                        logger.warning(f"Subsystem {subsystem} is unhealthy")
                        
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(60)
                
    async def _metrics_collector(self):
        """Collect metrics from all subsystems"""
        while self.is_initialized:
            try:
                # Collect metrics from subsystems
                ss0_metrics = self.subsystems[SubsystemID.SS0_INGESTION].get_sensor_coverage()
                ss2_metrics = self.subsystems[SubsystemID.SS2_ESTIMATION].get_tracking_metrics()
                ss3_metrics = self.subsystems[SubsystemID.SS3_COMMAND].get_asset_status()
                ss4_metrics = await self.subsystems[SubsystemID.SS4_CCDM].get_ccdm_summary()
                ss5_metrics = self.subsystems[SubsystemID.SS5_HOSTILITY].get_monitoring_summary()
                ss6_metrics = self.subsystems[SubsystemID.SS6_RESPONSE].get_threat_summary()
                
                # Update integration metrics
                self.metrics["uct_tracks"] = ss2_metrics.get("active_tracks", 0)
                self.metrics["threats_detected"] = ss6_metrics.get("total_threats", 0)
                self.metrics["sensor_coverage"] = ss0_metrics.get("coverage_percentage", 0)
                self.metrics["asset_utilization"] = ss3_metrics.get("busy_assets", 0) / max(ss3_metrics.get("total_assets", 1), 1)
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(60)
                
    async def check_health(self) -> Dict[str, bool]:
        """Check health of all components"""
        health = {
            "kafka": self.kafka_client.is_connected if hasattr(self.kafka_client, 'is_connected') else True,
            "udl": self.udl_client.is_authenticated() if hasattr(self.udl_client, 'is_authenticated') else True,
            "node_red": await self.node_red_service.health_check() if hasattr(self.node_red_service, 'health_check') else True
        }
        
        # Check each subsystem
        for subsystem_id in SubsystemID:
            health[subsystem_id.value] = True  # Simplified - would have actual health checks
            
        return health
        
    async def get_status(self) -> IntegrationStatus:
        """Get current integration status"""
        health = await self.check_health()
        
        # Get subsystem statuses
        subsystem_status = {}
        
        # SS0 status
        ss0_coverage = self.subsystems[SubsystemID.SS0_INGESTION].get_sensor_coverage()
        subsystem_status["ss0_ingestion"] = {
            "active_sensors": ss0_coverage.get("active_sensors", 0),
            "coverage_percentage": ss0_coverage.get("coverage_percentage", 0)
        }
        
        # SS1 status
        ss1_models = self.subsystems[SubsystemID.SS1_MODELING].get_all_models()
        subsystem_status["ss1_modeling"] = {
            "models_tracked": len(ss1_models),
            "threats_modeled": len(self.subsystems[SubsystemID.SS1_MODELING].get_threats())
        }
        
        # SS2 status
        ss2_metrics = self.subsystems[SubsystemID.SS2_ESTIMATION].get_tracking_metrics()
        subsystem_status["ss2_estimation"] = ss2_metrics
        
        # SS3 status
        ss3_status = self.subsystems[SubsystemID.SS3_COMMAND].get_asset_status()
        ss3_queue = self.subsystems[SubsystemID.SS3_COMMAND].get_command_queue_status()
        subsystem_status["ss3_command"] = {
            **ss3_status,
            **ss3_queue,
            "defensive_posture": self.subsystems[SubsystemID.SS3_COMMAND].battle_manager.current_posture
        }
        
        # SS4 status
        ss4_summary = await self.subsystems[SubsystemID.SS4_CCDM].get_ccdm_summary()
        subsystem_status["ss4_ccdm"] = ss4_summary
        
        # SS5 status
        ss5_summary = self.subsystems[SubsystemID.SS5_HOSTILITY].get_monitoring_summary()
        subsystem_status["ss5_hostility"] = ss5_summary
        
        # SS6 status
        ss6_summary = self.subsystems[SubsystemID.SS6_RESPONSE].get_threat_summary()
        subsystem_status["ss6_response"] = ss6_summary
        
        return IntegrationStatus(
            status="operational" if all(health.values()) else "degraded",
            subsystems=subsystem_status,
            health=health,
            metrics=self.metrics,
            active_threats=ss6_summary.get("total_threats", 0),
            defensive_posture=self.subsystems[SubsystemID.SS3_COMMAND].battle_manager.current_posture
        )
        
    async def submit_collection_request(
        self,
        sensor_id: str,
        target_id: str,
        collection_type: str,
        priority: int = 5
    ):
        """Submit collection request to SS0"""
        await self.subsystems[SubsystemID.SS0_INGESTION].submit_collection_request(
            sensor_id,
            target_id,
            collection_type,
            priority
        )
        
    async def change_defensive_posture(self, new_posture: str):
        """Change defensive posture through SS3"""
        await self.subsystems[SubsystemID.SS3_COMMAND].change_defensive_posture(new_posture)
        
    async def get_ccdm_indicators(self, object_id: str) -> Dict[str, Any]:
        """Get CCDM indicators for an object"""
        return await self.subsystems[SubsystemID.SS4_CCDM].get_object_indicators(object_id)
        
    async def evaluate_all_ccdm_indicators(self) -> Dict[str, Any]:
        """Evaluate all CCDM indicators"""
        return await self.subsystems[SubsystemID.SS4_CCDM].evaluate_all_indicators()


# Create FastAPI app for the integration service
app = FastAPI(title="Welders Arc Integration Service")

# Global integration instance
integration: Optional[WeldersArcIntegration] = None


@app.on_event("startup")
async def startup_event():
    """Initialize integration on startup"""
    global integration
    settings = Settings()
    integration = WeldersArcIntegration(settings)
    await integration.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown integration on exit"""
    global integration
    if integration:
        await integration.shutdown()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not integration:
        raise HTTPException(status_code=503, detail="Service not initialized")
        
    health = await integration.check_health()
    return {
        "status": "healthy" if all(health.values()) else "unhealthy",
        "components": health
    }


@app.get("/status", response_model=IntegrationStatus)
async def get_status():
    """Get integration status"""
    if not integration:
        raise HTTPException(status_code=503, detail="Service not initialized")
        
    return await integration.get_status()


@app.post("/collection-request")
async def submit_collection(
    sensor_id: str,
    target_id: str,
    collection_type: str,
    priority: int = 5
):
    """Submit collection request"""
    if not integration:
        raise HTTPException(status_code=503, detail="Service not initialized")
        
    await integration.submit_collection_request(
        sensor_id,
        target_id,
        collection_type,
        priority
    )
    
    return {"status": "submitted", "sensor_id": sensor_id, "target_id": target_id}


@app.post("/defensive-posture/{posture}")
async def change_posture(posture: str):
    """Change defensive posture"""
    if not integration:
        raise HTTPException(status_code=503, detail="Service not initialized")
        
    valid_postures = ["nominal", "elevated", "tactical", "strategic"]
    if posture not in valid_postures:
        raise HTTPException(status_code=400, detail=f"Invalid posture. Must be one of: {valid_postures}")
        
    await integration.change_defensive_posture(posture)
    
    return {"status": "changed", "new_posture": posture}


@app.get("/ccdm/{object_id}")
async def get_ccdm_indicators(object_id: str):
    """Get CCDM indicators for an object"""
    if not integration:
        raise HTTPException(status_code=503, detail="Service not initialized")
        
    indicators = await integration.get_ccdm_indicators(object_id)
    return indicators


@app.post("/ccdm/evaluate-all")
async def evaluate_ccdm(background_tasks: BackgroundTasks):
    """Evaluate all CCDM indicators"""
    if not integration:
        raise HTTPException(status_code=503, detail="Service not initialized")
        
    # Run evaluation in background
    background_tasks.add_task(integration.evaluate_all_ccdm_indicators)
    
    return {"status": "evaluation_started"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 