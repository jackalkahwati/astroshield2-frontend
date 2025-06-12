"""
Standard Kafka Topic Definitions for Space Domain Awareness System
Based on official Welders Arc topic naming convention
"""

class StandardKafkaTopics:
    """Standard Kafka topic definitions following ss[0-6].category.subcategory pattern"""
    
    # SS0 - Data Ingestion Topics
    SS0_DATA_LAUNCH_DETECTION = "ss0.data.launch-detection"
    SS0_DATA_MANIFOLD_REQUEST = "ss0.data.manifold.request"
    SS0_DATA_MANIFOLD_RESPONSE = "ss0.data.manifold.response"
    SS0_DATA_RF_CHANGES = "ss0.data.rf-changes"
    SS0_DATA_RSO_CHARACTERIZATION = "ss0.data.rso-charcterization"  # Note: typo in original spec
    SS0_DATA_WEATHER_CONTRAILS = "ss0.data.weather.contrails"
    SS0_DATA_WEATHER_LAUNCH_SITE = "ss0.data.weather.launch-site"
    SS0_DATA_WEATHER_NEUTRAL_DENSITY = "ss0.data.weather.neutral-density"
    SS0_DATA_WEATHER_REALTIME_ORBITAL_DENSITY = "ss0.data.weather.realtime-orbital-density-predictions"
    SS0_DATA_WEATHER_REFLECTIVITY = "ss0.data.weather.reflectivity"
    SS0_DATA_WEATHER_TURBULENCE = "ss0.data.weather.turbulence"
    SS0_DATA_WEATHER_VTEC = "ss0.data.weather.vtec"
    SS0_DATA_WEATHER_WINDSHEAR = "ss0.data.weather.windshear"
    SS0_DATA_WEATHER_WINDSHEAR_JETSTREAM = "ss0.data.weather.windshear-jetstream-level"
    SS0_DATA_WEATHER_WINDSHEAR_LOW = "ss0.data.weather.windshear-low-level"
    SS0_LAUNCH_PREDICTION_WINDOW = "ss0.launch-prediction.launch-window"
    SS0_SENSOR_HEARTBEAT = "ss0.sensor.heartbeat"
    SS0_SYNTHETIC_DATA_GROUND_IMAGERY = "ss0.synthetic-data.ground-imagery"
    SS0_SYNTHETIC_DATA_SKY_IMAGERY = "ss0.synthetic-data.sky-imagery"
    SS0_WEATHER_LAUNCH_SITE = "ss0.weather.launch-site"
    
    # SS1 - Target Modeling Topics
    SS1_INDICATORS_CAPABILITIES_UPDATED = "ss1.indicators.capabilities-updated"
    SS1_OBJECT_ATTRIBUTE_UPDATE_REQUESTED = "ss1.object-attribute-update-requested"
    SS1_REQUEST_STATE_VECTOR_PREDICTION = "ss1.request.state-vector-prediction"
    SS1_RESPONSE_STATE_VECTOR_PREDICTION = "ss1.response.state-vector-prediction"
    SS1_TMDB_OBJECT_INSERTED = "ss1.tmdb.object-inserted"
    SS1_TMDB_OBJECT_UPDATED = "ss1.tmdb.object-updated"
    
    # SS2 - State Estimation Topics
    SS2_ANALYSIS_ASSOCIATION_MESSAGE = "ss2.analysis.association-message"
    SS2_ANALYSIS_SCORE_MESSAGE = "ss2.analysis.score-message"
    SS2_DATA_ELSET_BEST_STATE = "ss2.data.elset.best-state"
    SS2_DATA_ELSET_CATALOG_NOMINEE = "ss2.data.elset.catalog-nominee"
    SS2_DATA_ELSET_SGP4 = "ss2.data.elset.sgp4"
    SS2_DATA_ELSET_SGP4_XP = "ss2.data.elset.sgp4-xp"
    SS2_DATA_ELSET_UCT_CANDIDATE = "ss2.data.elset.uct-candidate"
    SS2_DATA_EPHEMERIS = "ss2.data.ephemeris"
    SS2_DATA_OBSERVATION_TRACK = "ss2.data.observation-track"
    SS2_DATA_OBSERVATION_TRACK_CORRELATED = "ss2.data.observation-track.correlated"
    SS2_DATA_OBSERVATION_TRACK_TRUE_UCT = "ss2.data.observation-track.true-uct"
    SS2_DATA_ORBIT_DETERMINATION = "ss2.data.orbit-determination"
    SS2_DATA_STATE_VECTOR = "ss2.data.state-vector"
    SS2_DATA_STATE_VECTOR_BEST_STATE = "ss2.data.state-vector.best-state"
    SS2_DATA_STATE_VECTOR_CATALOG_NOMINEE = "ss2.data.state-vector.catalog-nominee"
    SS2_DATA_STATE_VECTOR_UCT_CANDIDATE = "ss2.data.state-vector.uct-candidate"
    SS2_REQUEST_PROPAGATION = "ss2.request.propagation"
    SS2_REQUEST_STATE_RECOMMENDATION = "ss2.request.state-recommendation"
    SS2_REQUESTS_GENERIC_REQUEST = "ss2.requests.generic-request"
    SS2_RESPONSE_PROPAGATION = "ss2.response.propagation"
    SS2_RESPONSE_STATE_RECOMMENDATION = "ss2.response.state-recommendation"
    SS2_RESPONSES_GENERIC_RESPONSE = "ss2.responses.generic-response"
    SS2_SERVICE_EVENT = "ss2.service.event"
    SS2_SERVICE_HEARTBEAT = "ss2.service.heartbeat"
    
    # SS3 - Command & Control Topics
    SS3_DATA_ACCESSWINDOW = "ss3.data.accesswindow"
    SS3_DATA_DETECTIONPROBABILITY = "ss3.data.detectionprobability"
    
    # SS4 - CCDM Topics
    SS4_ATTRIBUTES_ORBITAL_ATTRIBUTION = "ss4.attributes.orbital-attribution"
    SS4_CCDM_CCDM_DB = "ss4.ccdm.ccdm-db"
    SS4_CCDM_OOI = "ss4.ccdm.ooi"
    SS4_INDICATORS_AMR_CHANGES = "ss4.indicators.amr-changes"
    SS4_INDICATORS_AMR_OOF = "ss4.indicators.amr-oof"
    SS4_INDICATORS_CLASS_ANALYSTS_DISAGREEMENT = "ss4.indicators.class-analysts-disagreement"
    SS4_INDICATORS_IMAGING_MANEUVERS_POL_VIOLATIONS = "ss4.indicators.imaging-maneauvers-pol-violations"  # Note: typo in original spec
    SS4_INDICATORS_MANEUVERS_DETECTED = "ss4.indicators.maneuvers-detected"
    SS4_INDICATORS_MANEUVERS_RF_POL_OOF = "ss4.indicators.maneuvers-rf-pol-oof"
    SS4_INDICATORS_OBJECT_HIGH_RADIATION = "ss4.indicators.object-in-high-radiation-environment"
    SS4_INDICATORS_OBJECT_UNOCCUPIED_ORBIT = "ss4.indicators.object-in-unoccupied-orbit"
    SS4_INDICATORS_OBJECT_MANEUVERED_COVERAGE_GAPS = "ss4.indicators.object-maneuvered-coverage-gaps"
    SS4_INDICATORS_OBJECT_NOT_IN_UN_REGISTRY = "ss4.indicators.object-not-in-un-registry"
    SS4_INDICATORS_OBJECT_STABLE = "ss4.indicators.object-stable"
    SS4_INDICATORS_OBJECT_THREAT_FROM_KNOWN_SITE = "ss4.indicators.object-threat-from-known-site"
    SS4_INDICATORS_OPTICAL_RADAR_SIGNATURE_MISMATCH = "ss4.indicators.optical-radar-signature-mismatch"
    SS4_INDICATORS_OPTICAL_RADAR_SIGNATURE_OOF = "ss4.indicators.optical-radar-signature-oof"
    SS4_INDICATORS_ORBIT_OOF = "ss4.indicators.orbit-oof"
    SS4_INDICATORS_PARTNER_SYSTEM_STIMULATED_OBJECT = "ss4.indicators.partner-system-stimulated-object"
    SS4_INDICATORS_PROXIMITY_EVENTS_VALID_REMOTE_SENSE = "ss4.indicators.proximity-events-valid-remote-sense"
    SS4_INDICATORS_RF_DETECTED = "ss4.indicators.rf-detected"
    SS4_INDICATORS_STABILITY_CHANGED = "ss4.indicators.stability-changed"
    SS4_INDICATORS_SUB_SATS_DEPLOYED = "ss4.indicators.sub-sats-deployed"
    SS4_INDICATORS_TRACK_OBJECT_NUMBER_GREATER_THAN_LAUNCH = "ss4.indicators.track-object-number-greater-than-launch"
    SS4_INDICATORS_TRUE_UCT_IN_ECLIPSE = "ss4.indicators.true-uct-in-eclipse"
    SS4_INDICATORS_UNKNOWN_DEBRIS_SMA_HIGHER_THAN_PARENT = "ss4.indicators.unknown-debris-sma-higher-than-parent"
    SS4_INDICATORS_VALID_IMAGING_MANEUVERS = "ss4.indicators.valid-imaging-maneuvers"
    SS4_INDICATORS_VIOLATES_ITU_FCC_FILINGS = "ss4.indicators.violates-itu-fcc-filings"
    
    # SS5 - Hostility Monitoring Topics
    SS5_LAUNCH_ASAT_ASSESSMENT = "ss5.launch.asat-assessment"
    SS5_LAUNCH_COPLANAR_ASSESSMENT = "ss5.launch.coplanar-assessment"
    SS5_LAUNCH_COPLANAR_NOMINAL = "ss5.launch.coplanar-nominal"
    SS5_LAUNCH_COPLANAR_OPPORTUNITIES = "ss5.launch.coplanar-opportunities"
    SS5_LAUNCH_COPLANAR_PREDICTION = "ss5.launch.coplanar-prediction"
    SS5_LAUNCH_DETECTION = "ss5.launch.detection"
    SS5_LAUNCH_FUSED = "ss5.launch.fused"
    SS5_LAUNCH_INTENT_ASSESSMENT = "ss5.launch.intent-assessment"
    SS5_LAUNCH_NOMINAL = "ss5.launch.nominal"
    SS5_LAUNCH_PREDICTION = "ss5.launch.prediction"
    SS5_LAUNCH_TRACKLET = "ss5.launch.tracklet"
    SS5_LAUNCH_TRAJECTORY = "ss5.launch.trajectory"
    SS5_LAUNCH_WEATHER_CHECK = "ss5.launch.weather-check"
    SS5_ONORBIT_PRIORITY_OOI = "ss5.onorbit.priority-OOI"
    SS5_PEZ_WEZ_ANALYSIS_EO = "ss5.pez-wez-analysis.eo"
    SS5_PEZ_WEZ_PREDICTION_CONJUNCTION = "ss5.pez-wez-prediction.conjunction"
    SS5_PEZ_WEZ_PREDICTION_EO = "ss5.pez-wez-prediction.eo"
    SS5_PEZ_WEZ_PREDICTION_GRAPPLER = "ss5.pez-wez-prediction.grappler"
    SS5_PEZ_WEZ_PREDICTION_KKV = "ss5.pez-wez-prediction.kkv"
    SS5_PEZ_WEZ_PREDICTION_RF = "ss5.pez-wez-prediction.rf"
    SS5_PEZ_WEZ_INTENT_ASSESSMENT = "ss5.pez-wez.intent-assessment"
    SS5_POLYGON_CLOSURES = "ss5.polygon-closures"
    SS5_REENTRY_PREDICTION = "ss5.reentry.prediction"
    SS5_RPO_CLASSIFICATION = "ss5.rpo.classification"
    SS5_RPO_IMAGE = "ss5.rpo.image"
    SS5_RPO_INTENT = "ss5.rpo.intent"
    SS5_SEPARATION_DETECTION = "ss5.separation.detection"
    SS5_SERVICE_HEARTBEAT = "ss5.service.heartbeat"
    
    # SS6 - Threat Assessment Topics
    SS6_RESPONSE_RECOMMENDATION_LAUNCH = "ss6.response-recommendation.launch"
    SS6_RESPONSE_RECOMMENDATION_LAUNCH_ASAT = "ss6.response-recommendation.launch-asat-assessment"
    SS6_RESPONSE_RECOMMENDATION_LAUNCH_COPLANAR_RISK = "ss6.response-recommendation.launch-coplanar-risk"
    SS6_RESPONSE_RECOMMENDATION_LAUNCH_REENTRY = "ss6.response-recommendation.launch-reentry"
    SS6_RESPONSE_RECOMMENDATION_ON_ORBIT = "ss6.response-recommendation.on-orbit"
    SS6_RISK_MITIGATION_OPTIMAL_MANEUVER = "ss6.risk-mitigation.optimal-maneuver"
    SS6_SERVICE_HEARTBEAT = "ss6.service.heartbeat"
    
    # Test Environment Topics
    TEST_ENVIRONMENT_SS0 = "test-environment.ss0"
    TEST_ENVIRONMENT_SS1 = "test-environment.ss1"
    TEST_ENVIRONMENT_SS2 = "test-environment.ss2"
    TEST_ENVIRONMENT_SS3 = "test-environment.ss3"
    TEST_ENVIRONMENT_SS4 = "test-environment.ss4"
    TEST_ENVIRONMENT_SS5 = "test-environment.ss5"
    TEST_ENVIRONMENT_SS6 = "test-environment.ss6"
    
    # UI Topics
    UI_EVENT = "ui.event"
    
    # WA Topics (Welders Arc?)
    WA_NEW_KNOWN_OBJECT = "wa.new-known-object"
    WA_NEW_UNK_OBJECT = "wa.new-unk-object"
    
    @classmethod
    def get_all_topics(cls) -> list:
        """Get all topic names as a list"""
        topics = []
        for attr_name in dir(cls):
            if not attr_name.startswith('_') and attr_name.isupper():
                topic_name = getattr(cls, attr_name)
                if isinstance(topic_name, str):
                    topics.append(topic_name)
        return topics
    
    @classmethod
    def get_topics_by_subsystem(cls, subsystem: str) -> list:
        """Get all topics for a specific subsystem (e.g., 'SS0', 'SS1', etc.)"""
        topics = []
        prefix = f"{subsystem.lower()}_"
        for attr_name in dir(cls):
            if attr_name.startswith(prefix):
                topic_name = getattr(cls, attr_name)
                if isinstance(topic_name, str):
                    topics.append(topic_name)
        return topics


# Topic access mapping for AstroShield
# Based on the access matrix, AstroShield would be a producer/consumer
ASTROSHIELD_TOPIC_ACCESS = {
    # Read access (📖)
    "read": [
        StandardKafkaTopics.SS0_DATA_LAUNCH_DETECTION,
        StandardKafkaTopics.SS0_LAUNCH_PREDICTION_WINDOW,
        StandardKafkaTopics.SS0_SENSOR_HEARTBEAT,
        StandardKafkaTopics.SS1_INDICATORS_CAPABILITIES_UPDATED,
        StandardKafkaTopics.SS1_TMDB_OBJECT_INSERTED,
        StandardKafkaTopics.SS1_TMDB_OBJECT_UPDATED,
        StandardKafkaTopics.SS2_DATA_STATE_VECTOR_BEST_STATE,
        StandardKafkaTopics.SS3_DATA_ACCESSWINDOW,
        StandardKafkaTopics.SS3_DATA_DETECTIONPROBABILITY,
        StandardKafkaTopics.SS4_ATTRIBUTES_ORBITAL_ATTRIBUTION,
        StandardKafkaTopics.SS4_CCDM_CCDM_DB,
        StandardKafkaTopics.SS4_CCDM_OOI,
        StandardKafkaTopics.SS5_LAUNCH_DETECTION,
        StandardKafkaTopics.SS5_LAUNCH_FUSED,
        StandardKafkaTopics.SS5_LAUNCH_PREDICTION,
        StandardKafkaTopics.SS5_LAUNCH_TRAJECTORY,
    ],
    
    # Write access (📝)
    "write": [
        StandardKafkaTopics.SS5_LAUNCH_ASAT_ASSESSMENT,
        StandardKafkaTopics.SS5_LAUNCH_TRAJECTORY,
        StandardKafkaTopics.SS5_POLYGON_CLOSURES,
    ],
    
    # Both read/write access (🅱)
    "both": [
        StandardKafkaTopics.SS2_ANALYSIS_ASSOCIATION_MESSAGE,
        StandardKafkaTopics.SS2_DATA_ELSET_UCT_CANDIDATE,
        StandardKafkaTopics.SS2_DATA_OBSERVATION_TRACK,
        StandardKafkaTopics.SS2_DATA_OBSERVATION_TRACK_TRUE_UCT,
        StandardKafkaTopics.SS2_DATA_ORBIT_DETERMINATION,
        StandardKafkaTopics.SS2_DATA_STATE_VECTOR,
        StandardKafkaTopics.SS2_DATA_STATE_VECTOR_UCT_CANDIDATE,
        StandardKafkaTopics.SS4_INDICATORS_IMAGING_MANEUVERS_POL_VIOLATIONS,
        StandardKafkaTopics.SS4_INDICATORS_MANEUVERS_DETECTED,
        StandardKafkaTopics.SS5_LAUNCH_ASAT_ASSESSMENT,
        StandardKafkaTopics.SS5_LAUNCH_COPLANAR_ASSESSMENT,
        StandardKafkaTopics.SS5_REENTRY_PREDICTION,
        StandardKafkaTopics.SS6_RISK_MITIGATION_OPTIMAL_MANEUVER,
        StandardKafkaTopics.UI_EVENT,
    ]
} 