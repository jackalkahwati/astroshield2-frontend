"""
DnD Operations API Endpoints
Provides REST API for Dungeons and Dragons counter-CCD operations
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from ....sda_integration.dnd_integration import (
    DnDIntegrationService, BOGEYObject, BOGEYThreatLevel, CCDTactic
)
from ....sda_integration.udl_integration.udl_client import UDLClient
from ....sda_integration.kafka.kafka_client import WeldersArcKafkaClient
from ...dependencies import get_udl_client, get_kafka_client

router = APIRouter()


class BOGEYResponse(BaseModel):
    """BOGEY object response model"""
    dnd_id: str
    origin_object_id: str
    visual_magnitude: float
    rms_accuracy: float
    epoch: datetime
    position: List[float]
    velocity: List[float]
    threat_level: str
    suspected_tactics: List[str]
    confidence_score: float
    last_observation: Optional[datetime]
    observation_count: int


class BOGEYSummaryResponse(BaseModel):
    """BOGEY catalog summary response"""
    total_bogeys: int
    threat_levels: Dict[str, int]
    suspected_tactics: Dict[str, int]
    last_update: datetime


class ConjunctionAlert(BaseModel):
    """Conjunction alert model"""
    alert_type: str
    bogey_id: str
    protect_object_id: str
    protect_object_name: str
    risk_level: float
    threat_level: str
    suspected_tactics: List[str]
    estimated_tca: datetime
    recommended_actions: List[str]


class DebrisEventResponse(BaseModel):
    """Debris event response model"""
    parent_satno: str
    parent_intdes: str
    parent_name: str
    event_date: datetime
    total_objects: int
    cataloged_objects: int
    lost_objects: int
    dnd_found_lost: int
    current_uncatalogued: int


# Global DnD service instance
dnd_service: Optional[DnDIntegrationService] = None


async def get_dnd_service(
    udl_client: UDLClient = Depends(get_udl_client),
    kafka_client: WeldersArcKafkaClient = Depends(get_kafka_client)
) -> DnDIntegrationService:
    """Get or create DnD integration service"""
    global dnd_service
    if dnd_service is None:
        dnd_service = DnDIntegrationService(udl_client, kafka_client)
        await dnd_service.initialize()
    return dnd_service


@router.get("/bogeys", response_model=List[BOGEYResponse])
async def get_bogeys(
    threat_level: Optional[str] = Query(None, description="Filter by threat level"),
    tactic: Optional[str] = Query(None, description="Filter by suspected CCD tactic"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    service: DnDIntegrationService = Depends(get_dnd_service)
):
    """Get BOGEY objects from DnD catalog"""
    try:
        catalog = service.catalog_processor.bogey_catalog
        bogeys = list(catalog.values())
        
        # Apply filters
        if threat_level:
            threat_enum = BOGEYThreatLevel(threat_level.upper())
            bogeys = [b for b in bogeys if b.threat_level == threat_enum]
            
        if tactic:
            tactic_enum = CCDTactic(tactic.upper())
            bogeys = [b for b in bogeys if tactic_enum in b.suspected_tactics]
            
        # Limit results
        bogeys = bogeys[:limit]
        
        # Convert to response format
        response = []
        for bogey in bogeys:
            response.append(BOGEYResponse(
                dnd_id=bogey.dnd_id,
                origin_object_id=bogey.origin_object_id,
                visual_magnitude=bogey.visual_magnitude,
                rms_accuracy=bogey.rms_accuracy,
                epoch=bogey.epoch,
                position=bogey.position,
                velocity=bogey.velocity,
                threat_level=bogey.threat_level.value,
                suspected_tactics=[t.value for t in bogey.suspected_tactics],
                confidence_score=bogey.confidence_score,
                last_observation=bogey.last_observation,
                observation_count=bogey.observation_count
            ))
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get BOGEY objects: {str(e)}")


@router.get("/bogeys/{dnd_id}", response_model=BOGEYResponse)
async def get_bogey(
    dnd_id: str,
    service: DnDIntegrationService = Depends(get_dnd_service)
):
    """Get specific BOGEY object by DnD ID"""
    try:
        catalog = service.catalog_processor.bogey_catalog
        
        if dnd_id not in catalog:
            raise HTTPException(status_code=404, detail=f"BOGEY object {dnd_id} not found")
            
        bogey = catalog[dnd_id]
        
        return BOGEYResponse(
            dnd_id=bogey.dnd_id,
            origin_object_id=bogey.origin_object_id,
            visual_magnitude=bogey.visual_magnitude,
            rms_accuracy=bogey.rms_accuracy,
            epoch=bogey.epoch,
            position=bogey.position,
            velocity=bogey.velocity,
            threat_level=bogey.threat_level.value,
            suspected_tactics=[t.value for t in bogey.suspected_tactics],
            confidence_score=bogey.confidence_score,
            last_observation=bogey.last_observation,
            observation_count=bogey.observation_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get BOGEY object: {str(e)}")


@router.get("/bogeys/summary", response_model=BOGEYSummaryResponse)
async def get_bogey_summary(
    service: DnDIntegrationService = Depends(get_dnd_service)
):
    """Get summary of BOGEY catalog"""
    try:
        summary = await service.get_bogey_summary()
        
        return BOGEYSummaryResponse(
            total_bogeys=summary["totalBogeys"],
            threat_levels=summary["threatLevels"],
            suspected_tactics=summary["suspectedTactics"],
            last_update=datetime.fromisoformat(summary["lastUpdate"])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get BOGEY summary: {str(e)}")


@router.get("/debris-events", response_model=List[DebrisEventResponse])
async def get_debris_events(
    service: DnDIntegrationService = Depends(get_dnd_service)
):
    """Get known debris events"""
    try:
        events = service.catalog_processor.debris_events
        
        response = []
        for event in events.values():
            response.append(DebrisEventResponse(
                parent_satno=event.parent_satno,
                parent_intdes=event.parent_intdes,
                parent_name=event.parent_name,
                event_date=event.event_date,
                total_objects=event.total_objects,
                cataloged_objects=event.cataloged_objects,
                lost_objects=event.lost_objects,
                dnd_found_lost=event.dnd_found_lost,
                current_uncatalogued=event.current_uncatalogued
            ))
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get debris events: {str(e)}")


@router.post("/catalog/update")
async def update_catalog(
    background_tasks: BackgroundTasks,
    service: DnDIntegrationService = Depends(get_dnd_service)
):
    """Trigger manual catalog update"""
    try:
        background_tasks.add_task(service.catalog_processor.update_catalog)
        return {"message": "Catalog update initiated"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initiate catalog update: {str(e)}")


@router.post("/conjunctions/check")
async def check_conjunctions(
    background_tasks: BackgroundTasks,
    service: DnDIntegrationService = Depends(get_dnd_service)
):
    """Trigger manual conjunction check"""
    try:
        background_tasks.add_task(service.catalog_processor.check_protect_list_conjunctions)
        return {"message": "Conjunction check initiated"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initiate conjunction check: {str(e)}")


@router.get("/protect-list")
async def get_protect_list(
    service: DnDIntegrationService = Depends(get_dnd_service)
):
    """Get current protect list patterns"""
    try:
        protect_list = service.catalog_processor.protect_list
        return {
            "patterns": protect_list,
            "count": len(protect_list),
            "description": "High-value asset patterns for conjunction monitoring"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get protect list: {str(e)}")


@router.get("/statistics")
async def get_dnd_statistics(
    service: DnDIntegrationService = Depends(get_dnd_service)
):
    """Get DnD operational statistics"""
    try:
        catalog = service.catalog_processor.bogey_catalog
        debris_events = service.catalog_processor.debris_events
        
        # Calculate statistics
        total_bogeys = len(catalog)
        high_threat_bogeys = len([b for b in catalog.values() 
                                 if b.threat_level in [BOGEYThreatLevel.HIGH, BOGEYThreatLevel.CRITICAL]])
        
        # Recent activity (last 24 hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_bogeys = len([b for b in catalog.values() 
                           if b.last_observation and b.last_observation > recent_cutoff])
        
        # Debris statistics
        total_debris_objects = sum(event.current_uncatalogued for event in debris_events.values())
        
        # CCD tactic distribution
        tactic_counts = {}
        for bogey in catalog.values():
            for tactic in bogey.suspected_tactics:
                tactic_name = tactic.value
                tactic_counts[tactic_name] = tactic_counts.get(tactic_name, 0) + 1
        
        return {
            "catalog_statistics": {
                "total_bogeys": total_bogeys,
                "high_threat_bogeys": high_threat_bogeys,
                "recent_activity_24h": recent_bogeys,
                "threat_level_distribution": {
                    "critical": len([b for b in catalog.values() if b.threat_level == BOGEYThreatLevel.CRITICAL]),
                    "high": len([b for b in catalog.values() if b.threat_level == BOGEYThreatLevel.HIGH]),
                    "medium": len([b for b in catalog.values() if b.threat_level == BOGEYThreatLevel.MEDIUM]),
                    "low": len([b for b in catalog.values() if b.threat_level == BOGEYThreatLevel.LOW])
                }
            },
            "debris_statistics": {
                "known_events": len(debris_events),
                "total_uncatalogued_objects": total_debris_objects,
                "recent_events": len([e for e in debris_events.values() 
                                    if e.event_date > datetime.utcnow() - timedelta(days=180)])
            },
            "ccd_tactics": {
                "detected_tactics": tactic_counts,
                "most_common_tactic": max(tactic_counts.items(), key=lambda x: x[1])[0] if tactic_counts else None
            },
            "operational_status": {
                "last_catalog_update": datetime.utcnow().isoformat(),
                "protect_list_size": len(service.catalog_processor.protect_list),
                "system_status": "OPERATIONAL"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.get("/threat-assessment/{dnd_id}")
async def get_threat_assessment(
    dnd_id: str,
    service: DnDIntegrationService = Depends(get_dnd_service)
):
    """Get detailed threat assessment for a BOGEY object"""
    try:
        catalog = service.catalog_processor.bogey_catalog
        
        if dnd_id not in catalog:
            raise HTTPException(status_code=404, detail=f"BOGEY object {dnd_id} not found")
            
        bogey = catalog[dnd_id]
        
        # Calculate additional threat metrics
        altitude = (sum(x**2 for x in bogey.position)**0.5) - 6371.0  # km above Earth
        orbital_period = 2 * 3.14159 * ((sum(x**2 for x in bogey.position)**0.5)**3 / 398600.4418)**0.5 / 3600  # hours
        
        # Assess orbital regime
        if altitude < 2000:
            orbital_regime = "LEO"
        elif altitude < 35000:
            orbital_regime = "MEO"
        else:
            orbital_regime = "GEO"
            
        # Risk factors
        risk_factors = []
        if bogey.visual_magnitude > 18.0:
            risk_factors.append("DIM_SIGNATURE")
        if bogey.rms_accuracy > 5.0:
            risk_factors.append("POOR_TRACKING")
        if orbital_regime == "GEO":
            risk_factors.append("CRITICAL_ORBITAL_REGIME")
        if len(bogey.suspected_tactics) > 1:
            risk_factors.append("MULTIPLE_CCD_TACTICS")
            
        return {
            "dnd_id": dnd_id,
            "threat_assessment": {
                "threat_level": bogey.threat_level.value,
                "confidence_score": bogey.confidence_score,
                "risk_factors": risk_factors,
                "suspected_tactics": [t.value for t in bogey.suspected_tactics]
            },
            "orbital_characteristics": {
                "altitude_km": round(altitude, 2),
                "orbital_period_hours": round(orbital_period, 2),
                "orbital_regime": orbital_regime,
                "visual_magnitude": bogey.visual_magnitude,
                "tracking_accuracy_km": bogey.rms_accuracy
            },
            "operational_impact": {
                "conjunction_risk": "HIGH" if orbital_regime == "GEO" else "MEDIUM",
                "surveillance_priority": bogey.threat_level.value,
                "recommended_actions": service.catalog_processor._get_recommended_actions(bogey, 0.3)
            },
            "data_quality": {
                "last_observation": bogey.last_observation.isoformat() if bogey.last_observation else None,
                "observation_count": bogey.observation_count,
                "data_age_hours": (datetime.utcnow() - bogey.epoch).total_seconds() / 3600
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get threat assessment: {str(e)}")


# Add route tags for API documentation
router.tags = ["DnD Operations"] 