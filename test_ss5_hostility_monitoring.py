#!/usr/bin/env python3
"""
SDA SS5 Hostility Monitoring Test Suite
Tests all SS5 schemas and capabilities including official Kafka topics

Based on SDA Subsystem 5 documentation:
- Launch detection and intent assessment
- PEZ-WEZ predictions for threat engagement zones
- ASAT capability assessment
- Hostility monitoring and threat ranking
"""

import os
import sys
import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SS5HostilityMonitoringTestSuite:
    """Comprehensive test suite for SDA SS5 Hostility Monitoring capabilities"""
    
    def __init__(self):
        self.sda_integration = None
        self.test_results = {}
        
    async def setup(self) -> bool:
        """Setup SS5 test environment"""
        try:
            from asttroshield.sda_kafka import (
                SDAKafkaCredentials,
                AstroShieldSDAIntegration,
                SDATopicManager
            )
            
            # Initialize SDA integration in test mode
            credentials = SDAKafkaCredentials(
                bootstrap_servers="localhost:9092",
                username="test",
                password="test"
            )
            self.sda_integration = AstroShieldSDAIntegration(credentials)
            self.sda_integration.kafka_client.test_mode = True
            await self.sda_integration.initialize()
            
            logger.info("SS5 Hostility Monitoring test environment initialized")
            return True
            
        except Exception as e:
            logger.error(f"SS5 setup failed: {e}")
            return False
    
    async def test_ss5_topic_structure(self) -> bool:
        """Test SS5 Kafka topic structure and naming conventions"""
        try:
            from asttroshield.sda_kafka import SDATopicManager
            
            # Test all SS5 topics from official documentation
            ss5_topics = {
                "launch_asat_assessment": "ss5.launch.asat-assessment",
                "launch_coplanar_assessment": "ss5.launch.coplanar-assessment",
                "launch_coplanar_prediction": "ss5.launch.coplanar-prediction",
                "launch_detection": "ss5.launch.detection",
                "launch_intent_assessment": "ss5.launch.intent-assessment",
                "launch_nominal": "ss5.launch.nominal",
                "launch_prediction": "ss5.launch.prediction",
                "launch_trajectory": "ss5.launch.trajectory",
                "launch_weather_check": "ss5.launch.weather-check",
                "pez_wez_analysis_eo": "ss5.pez-wez-analysis.eo",
                "pez_wez_prediction_conjunction": "ss5.pez-wez-prediction.conjunction",
                "pez_wez_prediction_eo": "ss5.pez-wez-prediction.eo",
                "pez_wez_prediction_grappler": "ss5.pez-wez-prediction.grappler",
                "pez_wez_prediction_kkv": "ss5.pez-wez-prediction.kkv",
                "pez_wez_prediction_rf": "ss5.pez-wez-prediction.rf",
                "pez_wez_intent_assessment": "ss5.pez-wez.intent-assessment",
                "reentry_prediction": "ss5.reentry.prediction",
                "separation_detection": "ss5.separation.detection",
                "ss5_service_heartbeat": "ss5.service.heartbeat"
            }
            
            logger.info("Testing SS5 topic structure...")
            for topic_key, expected_topic in ss5_topics.items():
                actual_topic = SDATopicManager.get_topic(topic_key, use_test=False)
                assert actual_topic == expected_topic, f"Topic mismatch: {topic_key} -> {actual_topic} != {expected_topic}"
                logger.info(f"✓ {topic_key}: {actual_topic}")
            
            logger.info(f"✓ All {len(ss5_topics)} SS5 topics validated")
            return True
            
        except Exception as e:
            logger.error(f"SS5 topic structure test failed: {e}")
            return False
    
    async def test_launch_intent_assessment(self) -> bool:
        """Test SS5 launch intent assessment schema and publishing"""
        try:
            from asttroshield.sda_kafka.sda_schemas import SDASchemaFactory
            
            logger.info("Testing SS5 launch intent assessment...")
            
            # Create comprehensive intent assessment data
            intent_data = {
                "intent_category": "hostile",
                "threat_level": "high",
                "hostility_score": 0.87,
                "confidence": 0.92,
                "potential_targets": [
                    "GPS-III-SV01",
                    "STARLINK-1234", 
                    "ISS"
                ],
                "target_type": "satellite",
                "threat_indicators": [
                    "proximity_to_critical_assets",
                    "unusual_trajectory_profile",
                    "non_cooperative_behavior",
                    "weapon_payload_signatures"
                ],
                "asat_capability": True,
                "coplanar_threat": True,
                "analyst_id": "astroshield-threatscorer-1.0"
            }
            
            # Test schema creation
            sda_message = SDASchemaFactory.create_launch_intent_assessment(
                launch_id="LAUNCH-2024-001",
                source="astroshield",
                **intent_data
            )
            
            # Validate schema fields
            assert sda_message.source == "astroshield", "Source mismatch"
            assert sda_message.launchId == "LAUNCH-2024-001", "Launch ID mismatch"
            assert sda_message.intentCategory == "hostile", "Intent category mismatch"
            assert sda_message.threatLevel == "high", "Threat level mismatch"
            assert sda_message.hostilityScore == 0.87, "Hostility score mismatch"
            assert sda_message.asatCapability == True, "ASAT capability mismatch"
            assert sda_message.coplanarThreat == True, "Coplanar threat mismatch"
            assert len(sda_message.potentialTargets) == 3, "Potential targets count mismatch"
            assert len(sda_message.threatIndicators) == 4, "Threat indicators count mismatch"
            
            logger.info("✓ Schema validation passed")
            
            # Test publishing
            success = await self.sda_integration.publish_launch_intent_assessment(
                "LAUNCH-2024-001", 
                intent_data
            )
            
            assert success == True, "Publishing failed"
            logger.info("✓ SS5 launch intent assessment published successfully")
            
            # Test JSON serialization
            if hasattr(sda_message, 'json'):
                message_json = sda_message.json()
                parsed_data = json.loads(message_json)
                
                assert 'source' in parsed_data, "Missing source in JSON"
                assert 'launchId' in parsed_data, "Missing launchId in JSON"
                assert 'threatLevel' in parsed_data, "Missing threatLevel in JSON"
                assert 'hostilityScore' in parsed_data, "Missing hostilityScore in JSON"
                
                logger.info("✓ JSON serialization and structure valid")
                logger.info(f"   Intent: {parsed_data.get('intentCategory')}")
                logger.info(f"   Threat Level: {parsed_data.get('threatLevel')}")
                logger.info(f"   Hostility Score: {parsed_data.get('hostilityScore')}")
                logger.info(f"   Targets: {len(parsed_data.get('potentialTargets', []))}")
            
            return True
            
        except Exception as e:
            logger.error(f"Launch intent assessment test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_pez_wez_predictions(self) -> bool:
        """Test SS5 PEZ-WEZ prediction schemas for all weapon types"""
        try:
            from asttroshield.sda_kafka.sda_schemas import SDASchemaFactory
            
            logger.info("Testing SS5 PEZ-WEZ predictions...")
            
            # Test different weapon types as per SS5 topics
            weapon_types = ["eo", "rf", "kkv", "grappler", "conjunction"]
            
            for weapon_type in weapon_types:
                logger.info(f"Testing {weapon_type.upper()} weapon PEZ-WEZ prediction...")
                
                prediction_data = {
                    "weapon_type": weapon_type,
                    "pez_radius": 50.0,  # km
                    "wez_radius": 25.0,  # km
                    "engagement_probability": 0.85,
                    "time_to_engagement": 300.0,  # seconds
                    "engagement_window": [
                        datetime.now(timezone.utc) + timedelta(minutes=5),
                        datetime.now(timezone.utc) + timedelta(minutes=15)
                    ],
                    "target_assets": [
                        "GPS-III-SV01",
                        "STARLINK-1234"
                    ],
                    "primary_target": "GPS-III-SV01",
                    "validity_period": 2.0,  # hours
                    "confidence": 0.78
                }
                
                # Create schema
                sda_message = SDASchemaFactory.create_pez_wez_prediction(
                    threat_id=f"THREAT-{weapon_type.upper()}-001",
                    source="astroshield",
                    **prediction_data
                )
                
                # Validate schema
                assert sda_message.source == "astroshield", f"{weapon_type} source mismatch"
                assert sda_message.threatId == f"THREAT-{weapon_type.upper()}-001", f"{weapon_type} threat ID mismatch"
                assert sda_message.weaponType == weapon_type, f"{weapon_type} weapon type mismatch"
                assert sda_message.pezRadius == 50.0, f"{weapon_type} PEZ radius mismatch"
                assert sda_message.wezRadius == 25.0, f"{weapon_type} WEZ radius mismatch"
                assert sda_message.engagementProbability == 0.85, f"{weapon_type} engagement probability mismatch"
                
                # Test publishing
                success = await self.sda_integration.publish_pez_wez_prediction(
                    f"THREAT-{weapon_type.upper()}-001",
                    prediction_data
                )
                
                assert success == True, f"{weapon_type} publishing failed"
                logger.info(f"✓ {weapon_type.upper()} PEZ-WEZ prediction published successfully")
            
            logger.info(f"✓ All {len(weapon_types)} weapon types tested successfully")
            return True
            
        except Exception as e:
            logger.error(f"PEZ-WEZ prediction test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_asat_assessment(self) -> bool:
        """Test SS5 ASAT (Anti-Satellite) assessment schema"""
        try:
            from asttroshield.sda_kafka.sda_schemas import SDASchemaFactory
            
            logger.info("Testing SS5 ASAT assessment...")
            
            # Comprehensive ASAT assessment data
            assessment_data = {
                "asat_type": "kinetic",
                "asat_capability": True,
                "threat_level": "imminent",
                "targeted_assets": [
                    "GPS-III-SV01",
                    "GPS-III-SV02", 
                    "STARLINK-1234",
                    "ISS"
                ],
                "orbit_regimes_threatened": [
                    "MEO",
                    "LEO",
                    "GEO"
                ],
                "intercept_capability": True,
                "max_reach_altitude": 2000.0,  # km
                "effective_range": 1500.0,    # km
                "launch_to_impact": 15.5,     # minutes
                "confidence": 0.94,
                "intelligence_sources": [
                    "satellite_imagery",
                    "signals_intelligence",
                    "human_intelligence",
                    "trajectory_analysis"
                ]
            }
            
            # Create schema
            sda_message = SDASchemaFactory.create_asat_assessment(
                threat_id="ASAT-THREAT-001",
                source="astroshield",
                **assessment_data
            )
            
            # Validate schema fields
            assert sda_message.source == "astroshield", "ASAT source mismatch"
            assert sda_message.threatId == "ASAT-THREAT-001", "ASAT threat ID mismatch"
            assert sda_message.asatType == "kinetic", "ASAT type mismatch"
            assert sda_message.asatCapability == True, "ASAT capability mismatch"
            assert sda_message.threatLevel == "imminent", "ASAT threat level mismatch"
            assert sda_message.interceptCapability == True, "ASAT intercept capability mismatch"
            assert sda_message.maxReachAltitude == 2000.0, "ASAT max reach altitude mismatch"
            assert sda_message.effectiveRange == 1500.0, "ASAT effective range mismatch"
            assert sda_message.launchToImpact == 15.5, "ASAT launch to impact mismatch"
            assert len(sda_message.targetedAssets) == 4, "ASAT targeted assets count mismatch"
            assert len(sda_message.orbitRegimesThreatened) == 3, "ASAT orbit regimes count mismatch"
            assert len(sda_message.intelligence_sources) == 4, "ASAT intelligence sources count mismatch"
            
            logger.info("✓ ASAT schema validation passed")
            
            # Test publishing
            success = await self.sda_integration.publish_asat_assessment(
                "ASAT-THREAT-001",
                assessment_data
            )
            
            assert success == True, "ASAT publishing failed"
            logger.info("✓ SS5 ASAT assessment published successfully")
            
            # Test JSON output
            if hasattr(sda_message, 'json'):
                message_json = sda_message.json()
                parsed_data = json.loads(message_json)
                
                logger.info("✓ ASAT JSON serialization successful")
                logger.info(f"   ASAT Type: {parsed_data.get('asatType')}")
                logger.info(f"   Threat Level: {parsed_data.get('threatLevel')}")
                logger.info(f"   Max Reach: {parsed_data.get('maxReachAltitude')} km")
                logger.info(f"   Targets: {len(parsed_data.get('targetedAssets', []))}")
                logger.info(f"   Launch to Impact: {parsed_data.get('launchToImpact')} min")
            
            return True
            
        except Exception as e:
            logger.error(f"ASAT assessment test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_ss5_integration_workflow(self) -> bool:
        """Test complete SS5 workflow: Launch Detection -> Intent Assessment -> PEZ-WEZ -> ASAT"""
        try:
            logger.info("Testing complete SS5 hostility monitoring workflow...")
            
            # Step 1: Launch Detection (simulated - would come from SS5)
            launch_id = "HOSTILE-LAUNCH-2024-001"
            threat_id = "THREAT-KINETIC-001"
            
            logger.info(f"Step 1: Launch detected - {launch_id}")
            
            # Step 2: Intent Assessment
            intent_data = {
                "intent_category": "hostile",
                "threat_level": "high",
                "hostility_score": 0.89,
                "confidence": 0.95,
                "potential_targets": ["GPS-III-SV01", "STARLINK-1234"],
                "target_type": "satellite",
                "threat_indicators": [
                    "trajectory_analysis_hostile",
                    "non_cooperative_target",
                    "proximity_to_critical_assets"
                ],
                "asat_capability": True,
                "coplanar_threat": True,
                "analyst_id": "astroshield-threatscorer-1.0"
            }
            
            success_intent = await self.sda_integration.publish_launch_intent_assessment(
                launch_id, intent_data
            )
            assert success_intent, "Intent assessment failed"
            logger.info("Step 2: ✓ Intent assessment completed - HOSTILE threat identified")
            
            # Step 3: PEZ-WEZ Analysis for kinetic weapon
            pez_wez_data = {
                "weapon_type": "kkv",  # Kinetic Kill Vehicle
                "pez_radius": 75.0,
                "wez_radius": 35.0,
                "engagement_probability": 0.82,
                "time_to_engagement": 450.0,
                "engagement_window": [
                    datetime.now(timezone.utc) + timedelta(minutes=7),
                    datetime.now(timezone.utc) + timedelta(minutes=12)
                ],
                "target_assets": ["GPS-III-SV01"],
                "primary_target": "GPS-III-SV01",
                "validity_period": 1.5,
                "confidence": 0.88
            }
            
            success_pez_wez = await self.sda_integration.publish_pez_wez_prediction(
                threat_id, pez_wez_data
            )
            assert success_pez_wez, "PEZ-WEZ prediction failed"
            logger.info("Step 3: ✓ PEZ-WEZ analysis completed - Engagement zones calculated")
            
            # Step 4: ASAT Assessment
            asat_data = {
                "asat_type": "kinetic",
                "asat_capability": True,
                "threat_level": "imminent",
                "targeted_assets": ["GPS-III-SV01"],
                "orbit_regimes_threatened": ["MEO"],
                "intercept_capability": True,
                "max_reach_altitude": 20200.0,  # GPS altitude
                "effective_range": 1000.0,
                "launch_to_impact": 18.5,
                "confidence": 0.91,
                "intelligence_sources": [
                    "trajectory_analysis",
                    "payload_classification",
                    "threat_modeling"
                ]
            }
            
            success_asat = await self.sda_integration.publish_asat_assessment(
                threat_id, asat_data
            )
            assert success_asat, "ASAT assessment failed"
            logger.info("Step 4: ✓ ASAT assessment completed - IMMINENT threat to GPS constellation")
            
            # Workflow Summary
            logger.info("\n" + "="*60)
            logger.info("SS5 HOSTILITY MONITORING WORKFLOW COMPLETE")
            logger.info("="*60)
            logger.info(f"Launch ID: {launch_id}")
            logger.info(f"Threat ID: {threat_id}")
            logger.info(f"Intent: {intent_data['intent_category'].upper()}")
            logger.info(f"Threat Level: {intent_data['threat_level'].upper()}")
            logger.info(f"Hostility Score: {intent_data['hostility_score']}")
            logger.info(f"ASAT Capability: {asat_data['asat_capability']}")
            logger.info(f"Primary Target: {pez_wez_data['primary_target']}")
            logger.info(f"Engagement Probability: {pez_wez_data['engagement_probability']}")
            logger.info(f"Time to Impact: {asat_data['launch_to_impact']} minutes")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"SS5 integration workflow test failed: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run comprehensive SS5 test suite"""
        logger.info("🚀 Starting SDA SS5 Hostility Monitoring Test Suite")
        logger.info("="*70)
        
        tests = [
            ("SS5 Setup", self.setup),
            ("SS5 Topic Structure", self.test_ss5_topic_structure),
            ("Launch Intent Assessment", self.test_launch_intent_assessment),
            ("PEZ-WEZ Predictions", self.test_pez_wez_predictions),
            ("ASAT Assessment", self.test_asat_assessment),
            ("SS5 Integration Workflow", self.test_ss5_integration_workflow)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n🧪 Running test: {test_name}")
            try:
                success = await test_func()
                results[test_name] = success
                status = "PASS" if success else "FAIL"
                emoji = "✅" if success else "❌"
                logger.info(f"{emoji} {test_name}: {status}")
            except Exception as e:
                logger.error(f"❌ {test_name}: FAIL - {e}")
                results[test_name] = False
        
        return results
    
    async def cleanup(self):
        """Cleanup test resources"""
        if self.sda_integration:
            await self.sda_integration.stop()


async def main():
    """Run SS5 hostility monitoring tests"""
    test_suite = SS5HostilityMonitoringTestSuite()
    
    try:
        results = await test_suite.run_all_tests()
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("📊 SDA SS5 HOSTILITY MONITORING TEST RESULTS")
        logger.info("="*70)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, success in results.items():
            status = "PASS" if success else "FAIL"
            emoji = "✅" if success else "❌"
            logger.info(f"{emoji} {test_name}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All SS5 hostility monitoring tests PASSED!")
            logger.info("\n💡 Key SS5 Capabilities Verified:")
            logger.info("   ✅ Launch Intent Assessment (ss5.launch.intent-assessment)")
            logger.info("   ✅ PEZ-WEZ Predictions (ss5.pez-wez-prediction.*)")
            logger.info("   ✅ ASAT Assessment (ss5.launch.asat-assessment)")
            logger.info("   ✅ Threat Ranking and Hostility Scoring")
            logger.info("   ✅ Complete Threat Assessment Workflow")
            logger.info("\n🎯 AstroShield is ready for SDA SS5 integration!")
        else:
            logger.warning("⚠️  Some SS5 tests failed. Check output for details.")
        
        return passed == total
        
    finally:
        await test_suite.cleanup()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 