#!/usr/bin/env python3
"""
Comprehensive SDA Integration Test
Tests all subsystems: SS0, SS2, SS4, SS5, SS6

This test demonstrates the complete AstroShield integration with the SDA Welders Arc system.
"""

import sys
import os
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional

# Add project path
project_path = os.path.abspath('.')
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from src.asttroshield.sda_kafka.sda_message_bus import (
    SDAKafkaClient,
    SDAKafkaCredentials,
    AstroShieldSDAIntegration,
    SDATopicManager,
    SDASubsystem,
    MessagePriority
)

try:
    from src.asttroshield.sda_kafka.sda_schemas import SDASchemaFactory
    SDA_SCHEMAS_AVAILABLE = True
except ImportError:
    SDA_SCHEMAS_AVAILABLE = False
    print("SDA schemas not available - using fallback mode")

print(f"SDA Schemas Available: {SDA_SCHEMAS_AVAILABLE}")


class CompleteSDAIntegrationTest:
    """Complete SDA integration test across all subsystems"""
    
    def __init__(self):
        # Use mock credentials for testing
        self.credentials = SDAKafkaCredentials(
            bootstrap_servers="mock-kafka.sda.mil:9092",
            username="test_user",
            password="test_pass"
        )
        self.kafka_client = SDAKafkaClient(self.credentials, client_id="astroshield-complete-test")
        self.integration = AstroShieldSDAIntegration(self.credentials)
        
        # Test data
        self.test_object_id = "NORAD-12345"
        self.test_threat_id = "THREAT-2024-001"
        self.test_launch_id = "LAUNCH-2024-001"
        
    async def initialize(self):
        """Initialize SDA integration"""
        await self.kafka_client.initialize()
        await self.integration.initialize()
        print("✓ SDA integration initialized")
    
    async def test_ss0_data_ingestion(self):
        """Test SS0 Data Ingestion capabilities"""
        print("\n=== SS0 DATA INGESTION TEST ===")
        
        # Test 1: Lightning data
        lightning_result = await self.kafka_client.publish_weather_data(
            data_type="lightning_strike",
            timestamp=datetime.now(timezone.utc),
            weather_type="lightning",
            latitude=39.7392,
            longitude=-104.9903,
            value=145.0,
            units="kA",
            confidence=0.95
        )
        print(f"✓ Lightning data published: {lightning_result}")
        
        # Test 2: Orbital density data
        density_result = await self.kafka_client.publish_weather_data(
            data_type="neutral_density",
            timestamp=datetime.now(timezone.utc),
            weather_type="orbital_density",
            altitude=400.0,
            value=1.2e-12,
            units="kg/m³",
            confidence=0.88
        )
        print(f"✓ Orbital density data published: {density_result}")
        
        # Test 3: Weather forecast
        forecast_result = await self.kafka_client.publish_weather_data(
            data_type="turbulence_forecast",
            timestamp=datetime.now(timezone.utc),
            weather_type="turbulence",
            valid_time=datetime.now(timezone.utc) + timedelta(hours=6),
            forecast_time=datetime.now(timezone.utc) + timedelta(hours=12),
            latitude=45.0,
            longitude=-100.0,
            value=0.75,
            units="severity_index",
            confidence=0.82
        )
        print(f"✓ Weather forecast published: {forecast_result}")
        
        return lightning_result and density_result and forecast_result
    
    async def test_ss2_state_estimation(self):
        """Test SS2 State Estimation capabilities"""
        print("\n=== SS2 STATE ESTIMATION TEST ===")
        
        # Test 1: State vector publication
        state_vector_result = await self.kafka_client.publish_state_vector(
            object_id=self.test_object_id,
            position=[6778.137, 0.0, 0.0],  # km
            velocity=[0.0, 7.661, 0.0],      # km/s
            epoch=datetime.now(timezone.utc),
            topic_type="best_state",
            coordinate_frame="ITRF",
            data_source="astroshield_tracker",
            quality_metric=0.92
        )
        print(f"✓ State vector published: {state_vector_result}")
        
        # Test 2: TLE/Elset publication
        elset_result = await self.kafka_client.publish_elset(
            object_id=self.test_object_id,
            epoch=datetime.now(timezone.utc),
            mean_motion=15.5,  # revs/day
            eccentricity=0.001,
            inclination=51.6,  # degrees
            arg_of_perigee=90.0,
            raan=0.0,
            mean_anomaly=270.0,
            topic_type="best_state",
            catalog_number="12345",
            classification="U",
            rcs_size="MEDIUM",
            object_type="PAYLOAD",
            data_source="astroshield_tle_analyzer",
            quality_metric=0.95
        )
        print(f"✓ Elset published: {elset_result}")
        
        # Test 3: UCT candidate state vector
        uct_result = await self.kafka_client.publish_state_vector(
            object_id="UCT-CANDIDATE-001",
            position=[7000.0, 1000.0, 500.0],
            velocity=[0.5, 7.5, 0.1],
            epoch=datetime.now(timezone.utc),
            topic_type="uct_candidate",
            quality_metric=0.65
        )
        print(f"✓ UCT candidate published: {uct_result}")
        
        return state_vector_result and elset_result and uct_result
    
    async def test_ss4_ccdm_maneuver_detection(self):
        """Test SS4 CCDM Maneuver Detection capabilities"""
        print("\n=== SS4 CCDM MANEUVER DETECTION TEST ===")
        
        # Create comprehensive maneuver data
        maneuver_data = {
            "satellite_id": self.test_object_id,
            "pre_position": [6778.137, 0.0, 0.0],
            "pre_velocity": [0.0, 7.661, 0.0],
            "post_position": [6798.137, 0.0, 0.0],
            "post_velocity": [0.0, 7.651, 0.0],
            "event_start_time": datetime.now(timezone.utc) - timedelta(minutes=30),
            "event_stop_time": datetime.now(timezone.utc) - timedelta(minutes=25),
            "confidence": 0.89,
            "maneuver_type": "orbit_raising",
            "delta_v_magnitude": 0.015,  # km/s
            "detection_method": "astroshield_ai"
        }
        
        # Publish maneuver detection
        maneuver_result = await self.integration.publish_maneuver_detection(
            satellite_id=self.test_object_id,
            maneuver_data=maneuver_data
        )
        print(f"✓ Maneuver detection published: {maneuver_result}")
        
        return maneuver_result
    
    async def test_ss5_hostility_monitoring(self):
        """Test SS5 Hostility Monitoring capabilities"""
        print("\n=== SS5 HOSTILITY MONITORING TEST ===")
        
        # Test 1: Launch intent assessment
        intent_data = {
            "launch_id": self.test_launch_id,
            "intent_category": "military",
            "threat_level": "HIGH",
            "hostility_score": 0.87,
            "confidence": 0.91,
            "potential_targets": ["GPS-III-SV01", "GPS-III-SV02"],
            "target_type": "navigation",
            "threat_indicators": ["trajectory_analysis", "launch_site_military"],
            "asat_capability": True,
            "coplanar_threat": True,
            "analyst_id": "astroshield-orbital-intelligence"
        }
        
        intent_result = await self.integration.publish_launch_intent_assessment(
            launch_id=self.test_launch_id,
            intent_data=intent_data
        )
        print(f"✓ Launch intent assessment published: {intent_result}")
        
        # Test 2: PEZ-WEZ prediction for multiple weapon types
        pez_wez_results = []
        
        weapon_types = ["eo", "rf", "kkv", "grappler", "conjunction"]
        for weapon_type in weapon_types:
            prediction_data = {
                "threat_id": f"{self.test_threat_id}-{weapon_type}",
                "weapon_type": weapon_type,
                "pez_radius": 100.0,  # km
                "wez_radius": 50.0,   # km
                "engagement_probability": 0.78,
                "time_to_engagement": 45.5,  # minutes
                "target_assets": ["GPS-III-SV01"],
                "primary_target": "GPS-III-SV01",
                "confidence": 0.85
            }
            
            result = await self.integration.publish_pez_wez_prediction(
                threat_id=f"{self.test_threat_id}-{weapon_type}",
                prediction_data=prediction_data
            )
            pez_wez_results.append(result)
            print(f"✓ PEZ-WEZ {weapon_type} prediction published: {result}")
        
        # Test 3: ASAT assessment
        asat_data = {
            "threat_id": self.test_threat_id,
            "asat_type": "kinetic",
            "asat_capability": True,
            "threat_level": "CRITICAL",
            "targeted_assets": ["GPS-III-SV01", "TDRS-M"],
            "orbit_regimes_threatened": ["MEO", "GEO"],
            "intercept_capability": True,
            "max_reach_altitude": 2000.0,  # km
            "effective_range": 1500.0,     # km
            "launch_to_impact": 18.5,      # minutes
            "confidence": 0.93,
            "intelligence_sources": ["trajectory_analysis", "launch_site_intel"]
        }
        
        asat_result = await self.integration.publish_asat_assessment(
            threat_id=self.test_threat_id,
            assessment_data=asat_data
        )
        print(f"✓ ASAT assessment published: {asat_result}")
        
        return intent_result and all(pez_wez_results) and asat_result
    
    async def test_ss6_response_recommendation(self):
        """Test SS6 Response Recommendation capabilities"""
        print("\n=== SS6 RESPONSE RECOMMENDATION TEST ===")
        
        # Test 1: Launch threat response
        launch_response_result = await self.kafka_client.publish_response_recommendation(
            threat_id=self.test_launch_id,
            threat_type="hostile_launch",
            threat_level="HIGH",
            threatened_assets=["GPS-III-SV01", "GPS-III-SV02"],
            primary_coa="ACTIVATE_DEFENSIVE_MEASURES",
            priority="URGENT",
            confidence=0.91,
            recommendation_type="launch",
            alternate_coas=[
                "ENHANCE_MONITORING",
                "COORDINATE_INTERNATIONAL_RESPONSE",
                "PREPARE_EVASIVE_MANEUVERS"
            ],
            tactics_and_procedures=[
                "Deploy additional tracking assets",
                "Increase orbital awareness posture",
                "Coordinate with allies"
            ],
            time_to_implement=15.0,  # minutes
            rationale="High probability kinetic ASAT launch detected targeting critical navigation assets",
            risk_assessment="HIGH - Immediate threat to GPS constellation",
            analyst_id="astroshield-response-ai"
        )
        print(f"✓ Launch response recommendation published: {launch_response_result}")
        
        # Test 2: On-orbit threat response
        orbit_response_result = await self.kafka_client.publish_response_recommendation(
            threat_id=self.test_threat_id,
            threat_type="proximity_maneuver",
            threat_level="MEDIUM",
            threatened_assets=["TDRS-M"],
            primary_coa="EXECUTE_AVOIDANCE_MANEUVER",
            priority="IMMEDIATE",
            confidence=0.87,
            recommendation_type="on_orbit",
            alternate_coas=[
                "ENHANCE_TRACKING",
                "ASSESS_INTENT"
            ],
            tactics_and_procedures=[
                "Calculate optimal avoidance trajectory",
                "Coordinate with mission operations",
                "Monitor threat object behavior"
            ],
            time_to_implement=30.0,  # minutes
            rationale="Threat object maneuvering into close proximity with critical communication satellite",
            risk_assessment="MEDIUM - Potential collision or interference threat",
            analyst_id="astroshield-tactical-ai"
        )
        print(f"✓ On-orbit response recommendation published: {orbit_response_result}")
        
        return launch_response_result and orbit_response_result
    
    async def test_complete_scenario(self):
        """Test complete threat scenario across all subsystems"""
        print("\n=== COMPLETE THREAT SCENARIO TEST ===")
        
        # Scenario: Hostile launch detected → State estimation → Threat assessment → Response
        scenario_results = []
        
        # 1. Weather monitoring indicates launch window
        weather_alert = await self.kafka_client.publish_weather_data(
            data_type="launch_window_assessment",
            timestamp=datetime.now(timezone.utc),
            weather_type="clouds",
            latitude=40.0,
            longitude=45.0,
            value=0.1,  # Low cloud cover
            units="coverage_fraction",
            confidence=0.95
        )
        scenario_results.append(weather_alert)
        print("1. ✓ Weather conditions support launch window")
        
        # 2. Launch detection and intent assessment
        intent_assessment = await self.integration.publish_launch_intent_assessment(
            launch_id="SCENARIO-LAUNCH-001",
            intent_data={
                "intent_category": "hostile",
                "threat_level": "CRITICAL",
                "hostility_score": 0.94,
                "confidence": 0.89,
                "potential_targets": ["GPS-III-SV01"],
                "asat_capability": True,
                "analyst_id": "astroshield-scenario"
            }
        )
        scenario_results.append(intent_assessment)
        print("2. ✓ Hostile launch intent assessed")
        
        # 3. State estimation for threat object
        threat_state = await self.kafka_client.publish_state_vector(
            object_id="THREAT-SCENARIO-001",
            position=[7200.0, 0.0, 0.0],
            velocity=[0.0, 7.5, 0.0],
            epoch=datetime.now(timezone.utc),
            topic_type="uct_candidate",
            quality_metric=0.78
        )
        scenario_results.append(threat_state)
        print("3. ✓ Threat object state estimated")
        
        # 4. PEZ-WEZ analysis
        pez_wez_analysis = await self.integration.publish_pez_wez_prediction(
            threat_id="THREAT-SCENARIO-001",
            prediction_data={
                "weapon_type": "kkv",
                "pez_radius": 150.0,
                "wez_radius": 75.0,
                "engagement_probability": 0.85,
                "time_to_engagement": 22.5,
                "target_assets": ["GPS-III-SV01"],
                "confidence": 0.88
            }
        )
        scenario_results.append(pez_wez_analysis)
        print("4. ✓ PEZ-WEZ engagement zones calculated")
        
        # 5. ASAT capability assessment
        asat_assessment = await self.integration.publish_asat_assessment(
            threat_id="THREAT-SCENARIO-001",
            assessment_data={
                "asat_type": "kinetic",
                "asat_capability": True,
                "threat_level": "IMMINENT",
                "targeted_assets": ["GPS-III-SV01"],
                "launch_to_impact": 22.5,
                "confidence": 0.91
            }
        )
        scenario_results.append(asat_assessment)
        print("5. ✓ ASAT capability confirmed")
        
        # 6. Response recommendation
        response_recommendation = await self.kafka_client.publish_response_recommendation(
            threat_id="THREAT-SCENARIO-001",
            threat_type="imminent_asat_attack",
            threat_level="CRITICAL",
            threatened_assets=["GPS-III-SV01"],
            primary_coa="EXECUTE_EMERGENCY_EVASIVE_MANEUVER",
            priority="IMMEDIATE",
            confidence=0.93,
            recommendation_type="on_orbit",
            time_to_implement=5.0,  # minutes
            rationale="Imminent kinetic ASAT attack confirmed - immediate evasive action required",
            risk_assessment="CRITICAL - Loss of GPS satellite imminent without action"
        )
        scenario_results.append(response_recommendation)
        print("6. ✓ Emergency response recommendation issued")
        
        return all(scenario_results)
    
    async def run_all_tests(self):
        """Run all SDA integration tests"""
        print("🚀 Starting Complete SDA Integration Test")
        print("=" * 60)
        
        test_results = {}
        
        try:
            # Initialize
            await self.initialize()
            
            # Run individual subsystem tests
            test_results["SS0_Data_Ingestion"] = await self.test_ss0_data_ingestion()
            test_results["SS2_State_Estimation"] = await self.test_ss2_state_estimation()
            test_results["SS4_CCDM_Maneuver"] = await self.test_ss4_ccdm_maneuver_detection()
            test_results["SS5_Hostility_Monitoring"] = await self.test_ss5_hostility_monitoring()
            test_results["SS6_Response_Recommendation"] = await self.test_ss6_response_recommendation()
            
            # Run complete scenario test
            test_results["Complete_Scenario"] = await self.test_complete_scenario()
            
        except Exception as e:
            print(f"❌ Test error: {e}")
            test_results["Error"] = str(e)
        
        # Print results
        print("\n" + "=" * 60)
        print("🏁 COMPLETE SDA INTEGRATION TEST RESULTS")
        print("=" * 60)
        
        total_tests = len([k for k in test_results.keys() if k != "Error"])
        passed_tests = sum(1 for v in test_results.values() if v is True)
        
        for test_name, result in test_results.items():
            if test_name == "Error":
                print(f"❌ {test_name}: {result}")
            else:
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{status} {test_name.replace('_', ' ')}")
        
        print(f"\nOverall Results: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED - Complete SDA integration successful!")
        else:
            print(f"⚠️  {total_tests - passed_tests} tests failed")
        
        # Print SDA topic summary
        print("\n" + "=" * 60)
        print("📡 SDA TOPIC COVERAGE SUMMARY")
        print("=" * 60)
        
        topics_tested = [
            "SS0: Weather data ingestion",
            "SS2: State vector and elset publishing",
            "SS4: Maneuver detection",
            "SS5: Launch intent, PEZ-WEZ, ASAT assessment",
            "SS6: Response recommendations",
            "Complete threat scenario workflow"
        ]
        
        for topic in topics_tested:
            print(f"✓ {topic}")
        
        total_topics = len(SDATopicManager.TOPICS)
        print(f"\nTotal SDA Topics Available: {total_topics}")
        print(f"Schema Support: {'Enabled' if SDA_SCHEMAS_AVAILABLE else 'Fallback Mode'}")
        
        return test_results


async def main():
    """Main test function"""
    test_runner = CompleteSDAIntegrationTest()
    results = await test_runner.run_all_tests()
    
    # Exit with appropriate code
    if all(v for k, v in results.items() if k != "Error" and isinstance(v, bool)):
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 