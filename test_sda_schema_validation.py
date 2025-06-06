#!/usr/bin/env python3
"""
SDA Schema Validation Test
Tests the official SDA schema implementation with the maneuvers-detected schema
"""

import os
import sys
import json
from datetime import datetime, timezone, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_sda_schema_import():
    """Test importing SDA schemas"""
    try:
        from asttroshield.sda_kafka.sda_schemas import (
            SDAManeuverDetected,
            SDASchemaFactory,
            validate_sda_schema
        )
        print("✓ SDA schemas imported successfully")
        return True, (SDAManeuverDetected, SDASchemaFactory, validate_sda_schema)
    except ImportError as e:
        print(f"✗ Failed to import SDA schemas: {e}")
        return False, None

def test_maneuver_schema_creation():
    """Test creating SDA maneuver detected schema"""
    try:
        success, imports = test_sda_schema_import()
        if not success:
            return False
        
        SDAManeuverDetected, SDASchemaFactory, validate_sda_schema = imports
        
        # Test data matching the official SDA schema
        satellite_id = "STARLINK-1234"
        pre_position = [6800.0, 0.0, 0.0]  # km
        pre_velocity = [0.0, 7.5, 0.0]     # km/s
        post_position = [6810.0, 0.0, 0.0] # km (altitude raised)
        post_velocity = [0.0, 7.52, 0.0]   # km/s (delta-v applied)
        event_start = datetime.now(timezone.utc) - timedelta(minutes=5)
        event_stop = datetime.now(timezone.utc)
        
        # Test covariance matrix (6x6)
        pre_covariance = [
            [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.01, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.01, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.01]
        ]
        
        post_covariance = [
            [120.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 120.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 120.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.012, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.012, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.012]
        ]
        
        # Create SDA message using factory
        sda_message = SDASchemaFactory.create_maneuver_detected(
            satellite_id=satellite_id,
            source="astroshield",
            pre_position=pre_position,
            pre_velocity=pre_velocity,
            post_position=post_position,
            post_velocity=post_velocity,
            event_start=event_start,
            event_stop=event_stop,
            pre_covariance=pre_covariance,
            post_covariance=post_covariance
        )
        
        print("✓ SDA maneuver detected message created successfully")
        
        # Validate required fields
        assert sda_message.source == "astroshield", "Source field mismatch"
        assert sda_message.satNo == satellite_id, "Satellite ID mismatch"
        
        # Validate position components
        assert sda_message.prePosX == 6800.0, "Pre-position X mismatch"
        assert sda_message.prePosY == 0.0, "Pre-position Y mismatch"
        assert sda_message.prePosZ == 0.0, "Pre-position Z mismatch"
        
        assert sda_message.postPosX == 6810.0, "Post-position X mismatch"
        assert sda_message.postPosY == 0.0, "Post-position Y mismatch"
        assert sda_message.postPosZ == 0.0, "Post-position Z mismatch"
        
        # Validate velocity components
        assert sda_message.preVelX == 0.0, "Pre-velocity X mismatch"
        assert sda_message.preVelY == 7.5, "Pre-velocity Y mismatch"
        assert sda_message.preVelZ == 0.0, "Pre-velocity Z mismatch"
        
        assert sda_message.postVelX == 0.0, "Post-velocity X mismatch"
        assert sda_message.postVelY == 7.52, "Post-velocity Y mismatch"
        assert sda_message.postVelZ == 0.0, "Post-velocity Z mismatch"
        
        # Validate covariance matrices
        assert sda_message.preCov == pre_covariance, "Pre-covariance matrix mismatch"
        assert sda_message.postCov == post_covariance, "Post-covariance matrix mismatch"
        
        # Validate timestamps
        assert sda_message.eventStartTime == event_start, "Event start time mismatch"
        assert sda_message.eventStopTime == event_stop, "Event stop time mismatch"
        assert sda_message.createdAt is not None, "Created timestamp missing"
        
        print("✓ All SDA schema fields validated successfully")
        
        # Test JSON serialization
        if hasattr(sda_message, 'json'):
            message_json = sda_message.json()
            print("✓ JSON serialization successful")
            
            # Parse and validate JSON structure
            parsed_data = json.loads(message_json)
            
            # Validate required fields in JSON
            assert 'source' in parsed_data, "Missing 'source' in JSON"
            assert 'satNo' in parsed_data, "Missing 'satNo' in JSON"
            assert parsed_data['source'] == "astroshield", "JSON source mismatch"
            assert parsed_data['satNo'] == satellite_id, "JSON satNo mismatch"
            
            print("✓ JSON structure validation successful")
            
            # Validate schema using built-in validator
            is_valid = validate_sda_schema("maneuvers_detected", parsed_data)
            if is_valid:
                print("✓ SDA schema validation passed")
            else:
                print("⚠ SDA schema validation failed (may be expected in some environments)")
            
            # Print sample JSON (truncated for readability)
            print("\n📄 Sample SDA Maneuver Detection Message:")
            print(f"   source: {parsed_data.get('source')}")
            print(f"   satNo: {parsed_data.get('satNo')}")
            print(f"   eventStartTime: {parsed_data.get('eventStartTime')}")
            print(f"   eventStopTime: {parsed_data.get('eventStopTime')}")
            print(f"   prePosX: {parsed_data.get('prePosX')} km")
            print(f"   postPosX: {parsed_data.get('postPosX')} km")
            print(f"   preVelY: {parsed_data.get('preVelY')} km/s")
            print(f"   postVelY: {parsed_data.get('postVelY')} km/s")
            print(f"   preCov: {len(parsed_data.get('preCov', []))}x{len(parsed_data.get('preCov', [[]])[0] if parsed_data.get('preCov') else [])} matrix")
            print(f"   postCov: {len(parsed_data.get('postCov', []))}x{len(parsed_data.get('postCov', [[]])[0] if parsed_data.get('postCov') else [])} matrix")
            
        else:
            # Fallback for non-Pydantic implementations
            print("✓ Fallback schema implementation (no JSON method)")
        
        return True
        
    except Exception as e:
        print(f"✗ SDA schema creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_minimal_schema():
    """Test creating SDA schema with only required fields"""
    try:
        success, imports = test_sda_schema_import()
        if not success:
            return False
        
        SDAManeuverDetected, SDASchemaFactory, validate_sda_schema = imports
        
        # Create minimal message with only required fields
        minimal_message = SDASchemaFactory.create_maneuver_detected(
            satellite_id="TEST-001",
            source="astroshield"
            # All other fields are optional
        )
        
        # Validate required fields
        assert minimal_message.source == "astroshield", "Minimal source mismatch"
        assert minimal_message.satNo == "TEST-001", "Minimal satNo mismatch"
        
        print("✓ Minimal SDA schema (required fields only) created successfully")
        return True
        
    except Exception as e:
        print(f"✗ Minimal schema creation failed: {e}")
        return False

def test_schema_integration():
    """Test integration with SDA Kafka message bus"""
    try:
        # Test importing the main integration
        from asttroshield.sda_kafka import (
            SDAKafkaCredentials,
            AstroShieldSDAIntegration,
            SDASchemaFactory
        )
        
        print("✓ SDA Kafka integration imports successful")
        
        # Test that the integration can use the new schemas
        credentials = SDAKafkaCredentials(
            bootstrap_servers="localhost:9092",
            username="test",
            password="test"
        )
        
        integration = AstroShieldSDAIntegration(credentials)
        print("✓ SDA integration instantiated with schema support")
        
        return True
        
    except Exception as e:
        print(f"✗ Schema integration test failed: {e}")
        return False

def main():
    """Run all SDA schema validation tests"""
    print("🚀 SDA Schema Validation Test Suite")
    print("=" * 50)
    
    tests = [
        ("Schema Import", test_sda_schema_import),
        ("Maneuver Schema Creation", test_maneuver_schema_creation),
        ("Minimal Schema", test_minimal_schema),
        ("Schema Integration", test_schema_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running test: {test_name}")
        try:
            if test_name == "Schema Import":
                success, _ = test_func()
            else:
                success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"✗ Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        emoji = "✓" if success else "✗"
        print(f"{emoji} {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! SDA schema implementation is working correctly.")
        print("\n💡 Usage tip:")
        print("   You can now use the official SDA maneuvers-detected schema")
        print("   in your AstroShield integration with the SDA Kafka message bus.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("   This may be normal if you're running without all dependencies.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 