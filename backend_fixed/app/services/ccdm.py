from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import random
import logging
import asyncio
import threading
import time
from functools import wraps
import redis
import os
import math

# Add new imports for singleton pattern
from app.db.session import get_db

# Add import for audit logging
from app.services.audit_logger import audit_log, AuditLogger

from app.models.ccdm import (
    ObjectAnalysisResponse,
    ShapeChangeResponse,
    ThermalSignatureResponse,
    PropulsiveCapabilityResponse,
    HistoricalAnalysis,
    PropulsionType,
    PropulsionMetrics,
    CCDMAssessment,
    AnomalyDetection,
    ObjectThreatAssessment,
    ThreatLevel,
    HistoricalAnalysisResponse,
    AnalysisResult,
    ObjectAnalysisRequest,
    ObjectThreatAssessment,
    HistoricalAnalysisRequest,
    ShapeChangeRequest,
    ShapeChangeDetection,
    HistoricalAnalysisPoint,
    ThreatAssessmentRequest
)

from app.models.ccdm_orm import (
    CCDMAnalysisORM,
    ThreatAssessmentORM,
    AnalysisResultORM,
    HistoricalAnalysisORM,
    HistoricalAnalysisPointORM,
    ShapeChangeORM,
    ShapeChangeDetectionORM
)

# Configure logging
logger = logging.getLogger(__name__)

# Constants for database retries
DB_MAX_RETRIES = int(os.environ.get("DB_MAX_RETRIES", "3"))
DB_RETRY_DELAY = float(os.environ.get("DB_RETRY_DELAY", "0.5"))  # seconds

# Cache configuration
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)
REDIS_ENABLED = os.environ.get("REDIS_ENABLED", "true").lower() == "true"
DEFAULT_CACHE_TTL = int(os.environ.get("DEFAULT_CACHE_TTL", "300"))  # 5 minutes

# Singleton instance storage
_instance = None
_instance_lock = threading.Lock()

def retry_on_db_error(max_retries=DB_MAX_RETRIES, delay=DB_RETRY_DELAY):
    """
    Decorator for retrying database operations on failure
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for retry in range(max_retries + 1):  # +1 for the initial attempt
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    # Only retry on database-related errors, not on logical errors
                    db_error = any(err in str(e).lower() for err in [
                        "database", "connection", "sql", "timeout", "deadlock"
                    ])
                    
                    if not db_error or retry >= max_retries:
                        logger.error(f"Database operation failed after {retry+1} attempts: {str(e)}")
                        raise
                    
                    # Log retry attempt
                    logger.warning(f"Database operation failed, retrying ({retry+1}/{max_retries}): {str(e)}")
                    time.sleep(delay * (2 ** retry))  # Exponential backoff
            
            # This should never happen, but just in case
            raise last_exception
        return wrapper
    return decorator

class CCDMService:
    """
    CCDM (Conjunction and Collision Detection and Mitigation) service for AstroShield
    
    This service provides methods for analyzing space objects, assessing threats,
    and detecting shape changes.
    
    This class uses the Singleton pattern to prevent multiple initializations.
    """
    
    def __new__(cls, db: Session = None):
        """
        Implement Singleton pattern to ensure only one instance exists
        
        Args:
            db: Database session (optional)
            
        Returns:
            The singleton instance of CCDMService
        """
        global _instance
        
        if db is None:
            db = next(get_db())
        
        with _instance_lock:
            if _instance is None:
                logger.info("Creating new CCDMService instance")
                instance = super(CCDMService, cls).__new__(cls)
                instance._initialized = False
                _instance = instance
            else:
                logger.debug("Reusing existing CCDMService instance")
                # If a new DB session is provided, update the existing instance
                if db is not None:
                    _instance.db = db
                
        return _instance
    
    def __init__(self, db: Session = None):
        """
        Initialize the CCDM service
        
        Args:
            db: Database session
        """
        # Only initialize once (prevent re-initialization of singleton)
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        if db is None:
            db = next(get_db())
            
        self.db = db
        self._init_cache()
        # Thread lock for thread safety
        self.lock = threading.RLock()
        self._initialized = True
        logger.info("CCDMService initialized")
    
    def _init_cache(self):
        """Initialize Redis cache connection if enabled"""
        self.cache_enabled = REDIS_ENABLED
        
        if not self.cache_enabled:
            logger.info("Caching is disabled")
            self.cache = None
            return
            
        try:
            # Initialize Redis connection
            self.cache = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                decode_responses=False,  # Keep binary data for serialization
                socket_timeout=2.0,      # Don't hang if Redis is down
                socket_connect_timeout=2.0
            )
            
            # Test connection
            self.cache.ping()
            logger.info(f"Cache connection established to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis cache: {str(e)}")
            self.cache = None
            self.cache_enabled = False
    
    @audit_log("analyze", "space_object")
    def analyze_object(self, request: ObjectAnalysisRequest) -> ObjectAnalysisResponse:
        """
        Analyze a space object based on its NORAD ID
        
        Args:
            request: Object analysis request containing NORAD ID and options
            
        Returns:
            ObjectAnalysisResponse: Analysis results for the object
        """
        # In a real implementation, this would call prediction models
        # and retrieve actual data about the object
        
        # For testing, we'll generate mock data
        results = []
        for i in range(3):
            results.append(
                AnalysisResult(
                    timestamp=datetime.utcnow() - timedelta(hours=i),
                    confidence=round(random.uniform(0.7, 0.95), 2),
                    threat_level=random.choice(list(ThreatLevel)),
                    details={
                        "component": f"subsystem-{i+1}",
                        "anomaly_score": round(random.uniform(0, 1), 2)
                    }
                )
            )
        
        satellite_data = self._get_satellite_data(request.norad_id)
        
        return ObjectAnalysisResponse(
            norad_id=request.norad_id,
            timestamp=datetime.utcnow(),
            analysis_results=results,
            summary=f"Analysis completed for object {request.norad_id}",
            metadata=satellite_data
        )
    
    @audit_log("assess_threat", "space_object")
    def assess_threat(self, request: ThreatAssessmentRequest) -> ObjectThreatAssessment:
        """
        Assess the threat level of a space object
        
        Args:
            request: Threat assessment request containing NORAD ID and factors
            
        Returns:
            ObjectThreatAssessment: Threat assessment for the object
        """
        # In a real implementation, this would call threat assessment models
        
        # For testing, we'll generate mock threat data
        threat_components = {}
        for factor in request.assessment_factors:
            threat_components[factor.lower()] = random.choice(list(ThreatLevel)).__str__()
        
        # Determine overall threat level based on components
        threat_levels = [ThreatLevel(level) for level in threat_components.values() if level != "NONE"]
        overall_threat = max(threat_levels) if threat_levels else ThreatLevel.NONE
        
        recommendations = [
            "Monitor the object regularly",
            "Verify telemetry data with secondary sources",
            "Update trajectory predictions"
        ]
        
        if overall_threat == ThreatLevel.MEDIUM:
            recommendations.append("Consider potential evasive maneuvers")
        elif overall_threat in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            recommendations.append("Prepare for immediate evasive maneuvers")
            recommendations.append("Alert spacecraft operators")
        
        satellite_data = self._get_satellite_data(request.norad_id)
        
        return ObjectThreatAssessment(
            norad_id=request.norad_id,
            timestamp=datetime.utcnow(),
            overall_threat=overall_threat,
            confidence=round(random.uniform(0.7, 0.95), 2),
            threat_components=threat_components,
            recommendations=recommendations,
            metadata=satellite_data
        )
    
    @audit_log("access_historical_data", "space_object")
    def get_historical_analysis(
        self, 
        request: HistoricalAnalysisRequest, 
        page: int = 1, 
        page_size: int = 50
    ) -> HistoricalAnalysisResponse:
        """
        Get historical analysis data for a space object over a time period with pagination
        
        Args:
            request: Historical analysis request with NORAD ID and date range
            page: Page number for paginated results (1-based)
            page_size: Number of data points per page
            
        Returns:
            HistoricalAnalysisResponse: Historical analysis data
        """
        logger.info(f"Getting historical analysis for NORAD ID {request.norad_id} (page {page}, page_size {page_size})")
        
        # Add additional audit details
        AuditLogger.set_context(
            period_start=request.start_date.isoformat(),
            period_end=request.end_date.isoformat(),
            pagination=f"page={page},size={page_size}"
        )
        
        try:
            # Check if we have cached data for this specific page
            cache_key = f"historical:{request.norad_id}:{request.start_date.isoformat()}:{request.end_date.isoformat()}:{page}:{page_size}"
            cached_data = self._get_from_cache(cache_key)
            
            if cached_data:
                logger.info(f"Retrieved historical analysis from cache for NORAD ID {request.norad_id}")
                return cached_data
            
            # Try to get data from database with pagination
            db_data = self._get_historical_analysis_from_db(request, page, page_size)
            if db_data:
                # Cache the data before returning
                self._set_in_cache(cache_key, db_data, ttl=DEFAULT_CACHE_TTL)
                return db_data
            
            # Generate historical data if not in database
            days_diff = (request.end_date - request.start_date).days
            logger.info(f"Generating historical data for {days_diff+1} days for NORAD ID {request.norad_id}")
            
            # Get satellite metadata
            metadata = self._get_satellite_data(request.norad_id)
            
            # Generate all data points (need to do this to get total count for pagination)
            all_data_points = self._generate_historical_data_points(request)
            
            # Calculate pagination parameters
            total_points = len(all_data_points)
            total_pages = math.ceil(total_points / page_size)
            
            # Validate page number
            if page > total_pages and total_pages > 0:
                page = total_pages
            
            # Apply pagination
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, total_points)
            paginated_points = all_data_points[start_idx:end_idx]
            
            # Create response with pagination metadata
            response = HistoricalAnalysisResponse(
                norad_id=request.norad_id,
                start_date=request.start_date,
                end_date=request.end_date,
                analysis_points=paginated_points,
                trend_summary=f"Historical analysis for {days_diff} days shows normal behavior",
                metadata={
                    **metadata,
                    "pagination": {
                        "page": page,
                        "page_size": page_size,
                        "total_items": total_points,
                        "total_pages": total_pages
                    }
                }
            )
            
            # Cache the response
            self._set_in_cache(cache_key, response, ttl=DEFAULT_CACHE_TTL)
            
            # Store all data points in database asynchronously
            self._store_historical_data_async(request.norad_id, all_data_points)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in get_historical_analysis: {str(e)}", exc_info=True)
            # Re-raise as appropriate error type
            if "database connection" in str(e).lower():
                raise DatabaseError(f"Database connection error: {str(e)}", operation="get_historical_analysis")
            elif "no such satellite" in str(e).lower():
                raise ResourceNotFoundError("Satellite", str(request.norad_id))
            else:
                raise StandardError(
                    code=ErrorCode.SERVER_ERROR,
                    message=f"Error retrieving historical analysis: {str(e)}",
                    status_code=500
                )
    
    @retry_on_db_error()
    def _get_historical_analysis_from_db(
        self, 
        request: HistoricalAnalysisRequest,
        page: int = 1,
        page_size: int = 50
    ) -> Optional[HistoricalAnalysisResponse]:
        """
        Get historical analysis data from database with pagination
        
        Args:
            request: Historical analysis request with NORAD ID and date range
            page: Page number for paginated results (1-based)
            page_size: Number of data points per page
            
        Returns:
            Optional[HistoricalAnalysisResponse]: Historical analysis data or None if not found
        """
        conn = None
        try:
            conn = self.db.connection()
            cursor = conn.cursor()
            
            # First, get the total count of matching records
            count_query = """
                SELECT COUNT(*)
                FROM historical_analysis
                WHERE norad_id = %s
                AND analysis_date BETWEEN %s AND %s
            """
            cursor.execute(count_query, (request.norad_id, request.start_date, request.end_date))
            total_count = cursor.fetchone()[0]
            
            if total_count == 0:
                return None
                
            # Calculate pagination parameters
            total_pages = math.ceil(total_count / page_size)
            if page > total_pages:
                page = total_pages
                
            # Calculate offset
            offset = (page - 1) * page_size
            
            # Query for paginated historical data points
            query = """
                SELECT analysis_date, threat_level, confidence, data
                FROM historical_analysis
                WHERE norad_id = %s
                AND analysis_date BETWEEN %s AND %s
                ORDER BY analysis_date
                LIMIT %s OFFSET %s
            """
            cursor.execute(query, (
                request.norad_id, 
                request.start_date, 
                request.end_date,
                page_size,
                offset
            ))
            
            rows = cursor.fetchall()
            
            # Convert database rows to analysis points
            analysis_points = []
            for row in rows:
                analysis_date, threat_level, confidence, data = row
                analysis_points.append(
                    HistoricalAnalysisPoint(
                        timestamp=analysis_date,
                        threat_level=ThreatLevel(threat_level.upper()),
                        confidence=confidence,
                        details=data
                    )
                )
            
            # Get satellite metadata
            metadata = self._get_satellite_data(request.norad_id)
            
            # Add pagination metadata
            metadata["pagination"] = {
                "page": page,
                "page_size": page_size,
                "total_items": total_count,
                "total_pages": total_pages
            }
            
            # Create response
            return HistoricalAnalysisResponse(
                norad_id=request.norad_id,
                start_date=request.start_date,
                end_date=request.end_date,
                analysis_points=analysis_points,
                trend_summary=f"Historical analysis retrieved from database (page {page} of {total_pages})",
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Error retrieving historical analysis from database: {str(e)}", exc_info=True)
            raise  # Re-raise for retry decorator
        finally:
            # Ensure connection is closed properly
            if conn:
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(f"Error closing database connection: {str(e)}")
    
    def _store_historical_data_async(self, norad_id: int, data_points: List[HistoricalAnalysisPoint]) -> None:
        """
        Store historical data points in database asynchronously
        
        Args:
            norad_id: NORAD ID of the satellite
            data_points: List of historical analysis points to store
        """
        # Use a thread to avoid blocking the response
        threading.Thread(
            target=self._store_historical_data,
            args=(norad_id, data_points),
            daemon=True
        ).start()
    
    @retry_on_db_error(max_retries=5)  # More retries for important storage operation
    def _store_historical_data(self, norad_id: int, data_points: List[HistoricalAnalysisPoint]) -> None:
        """
        Store historical data points in database with retry logic
        
        Args:
            norad_id: NORAD ID of the satellite
            data_points: List of historical analysis points to store
        """
        with self.lock:  # Thread safety for database operations
            conn = None
            try:
                conn = self.db.connection()
                cursor = conn.cursor()
                
                # Use a batch insert for better performance
                batch_size = 50
                for i in range(0, len(data_points), batch_size):
                    batch = data_points[i:i+batch_size]
                    
                    # Prepare the query with multiple value sets
                    values = []
                    for point in batch:
                        values.append((
                            norad_id,
                            point.timestamp,
                            point.threat_level.value.lower(),
                            point.confidence,
                            point.details
                        ))
                        
                    query = """
                        INSERT INTO historical_analysis 
                        (norad_id, analysis_date, threat_level, confidence, data)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (norad_id, analysis_date) DO UPDATE
                        SET threat_level = EXCLUDED.threat_level,
                            confidence = EXCLUDED.confidence,
                            data = EXCLUDED.data
                    """
                    
                    cursor.executemany(query, values)
                    
                conn.commit()
                logger.info(f"Stored {len(data_points)} historical data points for NORAD ID {norad_id}")
                
            except Exception as e:
                if conn:
                    try:
                        conn.rollback()
                    except:
                        pass
                logger.error(f"Error storing historical data: {str(e)}", exc_info=True)
                raise  # Re-raise for retry decorator
            finally:
                # Ensure connection is closed properly
                if conn:
                    try:
                        conn.close()
                    except Exception as e:
                        logger.warning(f"Error closing database connection: {str(e)}")
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """
        Get item from cache if available
        
        Args:
            key: Cache key
            
        Returns:
            Cached item or None if not found
        """
        if not self.cache_enabled or not self.cache:
            return None
        
        try:
            import pickle
            cached_data = self.cache.get(key)
            if cached_data:
                return pickle.loads(cached_data)
            return None
        except Exception as e:
            logger.warning(f"Error getting item from cache: {str(e)}")
            return None
        
    def _set_in_cache(self, key: str, value: Any, ttl: int = DEFAULT_CACHE_TTL) -> None:
        """
        Store item in cache
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Time to live in seconds (default: 5 minutes)
        """
        if not self.cache_enabled or not self.cache:
            return
        
        try:
            import pickle
            serialized_value = pickle.dumps(value)
            self.cache.setex(key, ttl, serialized_value)
            logger.debug(f"Stored item in cache with key {key} and TTL {ttl}s")
        except Exception as e:
            logger.warning(f"Error setting item in cache: {str(e)}")
    
    @audit_log("detect_shape_changes", "space_object")
    def detect_shape_changes(self, request: ShapeChangeRequest) -> ShapeChangeResponse:
        """
        Detect shape changes for a space object over a time period
        
        Args:
            request: Shape change request with NORAD ID and date range
            
        Returns:
            ShapeChangeResponse: Detected shape changes
        """
        # In a real implementation, this would analyze observation data
        # to detect changes in the object's shape
        
        # For testing, we'll generate mock shape change data
        changes = []
        num_changes = random.randint(0, 3)  # Random number of changes
        
        days_diff = (request.end_date - request.start_date).days
        
        for i in range(num_changes):
            change_day = random.randint(0, days_diff)
            change_date = request.start_date + timedelta(days=change_day)
            component = random.choice(['solar panel', 'antenna', 'main body', 'sensor array'])
            
            changes.append(
                ShapeChangeDetection(
                    timestamp=change_date,
                    description=f"Detected change in {component} configuration",
                    confidence=round(random.uniform(0.6, 0.9), 2),
                    before_shape="standard_configuration",
                    after_shape="modified_configuration",
                    significance=round(random.uniform(0.1, 0.8), 2)
                )
            )
        
        satellite_data = self._get_satellite_data(request.norad_id)
        
        summary = "No significant shape changes detected."
        if changes:
            summary = f"Detected {len(changes)} shape changes with average significance of {sum(c.significance for c in changes)/len(changes):.2f}"
        
        return ShapeChangeResponse(
            norad_id=request.norad_id,
            detected_changes=changes,
            summary=summary,
            metadata=satellite_data
        )
    
    @audit_log("quick_assess", "space_object")
    def quick_assess_norad_id(self, norad_id: int) -> ObjectThreatAssessment:
        """
        Quick threat assessment for a space object by NORAD ID
        
        Args:
            norad_id: NORAD ID of the space object
            
        Returns:
            ObjectThreatAssessment: Quick threat assessment
        """
        # Create a request object and call the assess_threat method
        request = ThreatAssessmentRequest(
            norad_id=norad_id,
            assessment_factors=["COLLISION", "MANEUVER", "DEBRIS"]
        )
        
        return self.assess_threat(request)
    
    @audit_log("access_weekly_analysis", "space_object")
    def get_last_week_analysis(self, norad_id: int) -> HistoricalAnalysisResponse:
        """
        Get analysis data for the last week for a space object
        
        Args:
            norad_id: NORAD ID of the space object
            
        Returns:
            HistoricalAnalysisResponse: Last week's analysis data
        """
        # Create a request object for the last week and call get_historical_analysis
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        
        request = HistoricalAnalysisRequest(
            norad_id=norad_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return self.get_historical_analysis(request)
    
    def _generate_historical_data_points(self, request: HistoricalAnalysisRequest) -> List[HistoricalAnalysisPoint]:
        """
        Generate historical data points for the request period
        
        Args:
            request: Historical analysis request
            
        Returns:
            List of historical analysis points
        """
        days_diff = (request.end_date - request.start_date).days
        points = []
        
        # In a real implementation, this would use actual algorithms
        # and historical data sources to generate meaningful analysis
        for i in range(days_diff + 1):
            current_date = request.start_date + timedelta(days=i)
            
            # Create a deterministic but varied threat level based on the date and satellite ID
            # to ensure consistent results for repeated calls
            seed_value = request.norad_id + current_date.day + current_date.month + current_date.year
            random.seed(seed_value)
            
            # Generate threat level with weighted probabilities
            threat_level_rand = random.random()
            if threat_level_rand < 0.6:  # 60% chance of NONE
                threat_level = ThreatLevel.NONE
            elif threat_level_rand < 0.85:  # 25% chance of LOW
                threat_level = ThreatLevel.LOW
            elif threat_level_rand < 0.95:  # 10% chance of MEDIUM
                threat_level = ThreatLevel.MEDIUM
            else:  # 5% chance of HIGH
                threat_level = ThreatLevel.HIGH
                
            # Generate a confidence value based on threat level
            # Higher confidence for NONE/LOW, lower for MEDIUM/HIGH
            if threat_level == ThreatLevel.NONE:
                confidence = round(random.uniform(0.85, 0.95), 2)
            elif threat_level == ThreatLevel.LOW:
                confidence = round(random.uniform(0.75, 0.90), 2)
            elif threat_level == ThreatLevel.MEDIUM:
                confidence = round(random.uniform(0.65, 0.85), 2)
            else:
                confidence = round(random.uniform(0.60, 0.80), 2)
                
            # Generate details with metrics relevant to the threat level
            details = {
                "day": i,
                "date": current_date.isoformat(),
                "metrics": {
                    "position_uncertainty": round(random.uniform(5, 50), 1),
                    "velocity_uncertainty": round(random.uniform(0.01, 0.1), 3),
                    "signal_strength": round(random.uniform(-95, -75), 1),
                    "maneuver_probability": round(random.uniform(0, 0.3), 2)
                }
            }
            
            # Add anomaly information for higher threat levels
            if threat_level != ThreatLevel.NONE:
                anomaly_count = random.randint(1, 3 if threat_level == ThreatLevel.HIGH else 2)
                details["anomalies"] = []
                
                for j in range(anomaly_count):
                    anomaly_type = random.choice(["position", "signal", "behavior", "trajectory"])
                    details["anomalies"].append({
                        "type": anomaly_type,
                        "severity": threat_level.value.lower(),
                        "confidence": round(random.uniform(confidence - 0.1, confidence + 0.1), 2)
                    })
            
            # Create analysis point
            points.append(
                HistoricalAnalysisPoint(
                    timestamp=current_date,
                    threat_level=threat_level,
                    confidence=confidence,
                    details=details
                )
            )
            
        # Reset the random seed
        random.seed()
        
        return points
    
    def _get_satellite_data(self, norad_id: int) -> Dict[str, Any]:
        """
        Get satellite data for the NORAD ID
        
        Args:
            norad_id: NORAD ID of the satellite
            
        Returns:
            Dictionary with satellite metadata
        """
        # In a real implementation, this would query a database or external service
        # For now, return mock data
        return {
            "norad_id": norad_id,
            "name": f"Satellite-{norad_id}",
            "international_designator": f"2023-{norad_id:03d}A",
            "orbit_type": random.choice(["LEO", "MEO", "GEO", "HEO"]),
            "apogee": random.randint(500, 36000),
            "perigee": random.randint(300, 1000),
            "period_minutes": random.randint(90, 1440),
            "launch_date": (datetime.utcnow() - timedelta(days=random.randint(30, 3650))).isoformat(),
            "country": random.choice(["USA", "RUS", "CHN", "ESA", "JPN", "IND"]),
            "status": random.choice(["ACTIVE", "INACTIVE", "DECOMMISSIONED"])
        }

def get_ccdm_service(db: Session = None) -> CCDMService:
    """
    Factory function to get the singleton CCDM service instance.
    
    Args:
        db: Database session (optional)
        
    Returns:
        The singleton CCDMService instance
    """
    return CCDMService(db)