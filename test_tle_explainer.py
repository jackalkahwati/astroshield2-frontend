"""
Test script for TLE explainer
Run this to test the TLE explainer functionality locally
"""

import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, 'backend_fixed')

from app.ml_infrastructure.tle_explainer import TLEExplainerInput, MockTLEExplainerService

# Test with both mock and HF model if available
try:
    from app.ml_infrastructure.tle_explainer_hf import TLEExplainerServiceHF
    HF_AVAILABLE = True
except ImportError as e:
    print(f"Hugging Face model not available: {e}")
    HF_AVAILABLE = False

async def test_tle_explainer():
    """Test the TLE explainer with sample data"""
    
    # Sample TLE data
    test_cases = [
        {
            "name": "ISS (ZARYA)",
            "norad_id": "25544",
            "line1": "1 25544U 98067A   24079.07757601 .00016717 00000+0 10270-3 0  9994",
            "line2": "2 25544  51.6400 337.6640 0007776  35.5310 330.5120 15.50377579499263"
        },
        {
            "name": "GPS BIIR-2 (PRN 13)",
            "norad_id": "28474",
            "line1": "1 28474U 04045A   24079.91667824 -.00000035  00000-0  00000-0 0  9991",
            "line2": "2 28474  54.1047 140.0123 0163006 231.7851 126.3093  2.00570142147821"
        },
        {
            "name": "STARLINK-1007",
            "norad_id": "44713",
            "line1": "1 44713U 19074A   24079.54166667  .00002182  00000-0  15767-3 0  9992",
            "line2": "2 44713  53.0539 339.0123 0001361  88.8123 271.3123 15.06378579252123"
        }
    ]
    
    print("🛰️  Testing TLE Orbit Explainer")
    print("=" * 50)
    
    # Test with mock service
    print("\n📊 Testing with Mock Service:")
    mock_service = MockTLEExplainerService()
    
    for test_case in test_cases:
        print(f"\n🔍 Analyzing {test_case['name']} (NORAD {test_case['norad_id']}):")
        
        tle_input = TLEExplainerInput(
            norad_id=test_case['norad_id'],
            satellite_name=test_case['name'],
            line1=test_case['line1'],
            line2=test_case['line2']
        )
        
        try:
            explanation = await mock_service.explain_tle(tle_input)
            
            print(f"  📝 Description: {explanation.orbit_description}")
            print(f"  🌍 Orbit Type: {explanation.orbit_type}")
            print(f"  📏 Altitude: {explanation.altitude_description}")
            print(f"  ⏰ Period: {explanation.period_minutes:.1f} minutes")
            print(f"  📐 Inclination: {explanation.inclination_degrees:.1f}°")
            print(f"  ⚠️  Decay Risk: {explanation.decay_risk_level} ({explanation.decay_risk_score:.3f})")
            
            if explanation.anomaly_flags:
                print(f"  🚨 Anomalies: {', '.join(explanation.anomaly_flags)}")
            else:
                print(f"  ✅ No anomalies detected")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Test with HF model if available
    if HF_AVAILABLE and os.getenv("TEST_HUGGINGFACE", "false").lower() == "true":
        print("\n🤖 Testing with Hugging Face Model:")
        print("⚠️  This will download the model on first run (~15GB)")
        
        hf_service = TLEExplainerServiceHF()
        
        # Test with just ISS for HF model
        iss_case = test_cases[0]
        print(f"\n🔍 Analyzing {iss_case['name']} with AI model:")
        
        tle_input = TLEExplainerInput(
            norad_id=iss_case['norad_id'],
            satellite_name=iss_case['name'],
            line1=iss_case['line1'],
            line2=iss_case['line2']
        )
        
        try:
            explanation = await hf_service.explain_tle(tle_input)
            
            print(f"  📝 AI Description: {explanation.orbit_description}")
            print(f"  🤖 Model Output: {explanation.technical_details.get('model_output', 'N/A')}")
            print(f"  🎯 Confidence: {explanation.confidence_score:.3f}")
            
        except Exception as e:
            print(f"  ❌ HF Model Error: {e}")
    
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    asyncio.run(test_tle_explainer()) 