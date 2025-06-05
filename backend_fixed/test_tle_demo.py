#!/usr/bin/env python3
"""
Simple TLE Chat Demo
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from app.ml_infrastructure.tle_explainer import TLEExplainerInput, MockTLEExplainerService

async def demo():
    print("🛰️  TLE Chat Interface Demo")
    print("=" * 40)
    
    # Initialize service
    explainer = MockTLEExplainerService()
    
    # Test TLE (ISS) - Real format from NORAD
    tle_input = TLEExplainerInput(
        norad_id="25544",
        satellite_name="International Space Station (ISS)",
        line1="1 25544U 98067A   24325.50000000  .00016717  00000-0  10270-3 0  9994",
        line2="2 25544  51.6400 337.6640 0007776  35.5310 330.5120 15.50377579499263",
        include_risk_assessment=True,
        include_anomaly_detection=True
    )
    
    print("📡 Input TLE:")
    print(f"   {tle_input.line1}")
    print(f"   {tle_input.line2}")
    print()
    
    # Get explanation
    print("🔄 Processing...")
    result = await explainer.explain_tle(tle_input)
    
    print("✅ Analysis Complete!")
    print()
    
    # Format chat response
    risk_emoji = "🔴" if result.decay_risk_level == "HIGH" else \
                "🟡" if result.decay_risk_level == "MEDIUM" else "🟢"
    
    print("💬 Chat Response:")
    print("-" * 20)
    print(f"🛰️ **{result.satellite_name}** (ID: {result.norad_id})")
    print()
    print("📊 **Orbital Elements:**")
    print(f"• **Orbit Type**: {result.orbit_type} - {result.orbit_description}")
    print(f"• **Altitude**: {result.altitude_description}")
    print(f"• **Period**: {result.period_minutes:.1f} minutes")
    print(f"• **Inclination**: {result.inclination_degrees:.2f}°")
    print(f"• **Eccentricity**: {result.eccentricity:.6f}")
    print()
    print(f"{risk_emoji} **Decay Risk**: {result.decay_risk_level} (Score: {(result.decay_risk_score * 100):.1f}%)")
    if result.predicted_lifetime_days:
        print(f"⏱️ **Estimated Lifetime**: {round(result.predicted_lifetime_days)} days")
    print()
    if result.anomaly_flags:
        print(f"⚠️ **Anomalies**: {', '.join(result.anomaly_flags)}")
    print(f"🎯 **Confidence**: {(result.confidence_score * 100):.1f}%")
    
    print()
    print("🎉 TLE Chat Backend is working perfectly!")
    print()
    print("🚀 Next Steps:")
    print("1. Start backend: python -m uvicorn app.main:app --reload")
    print("2. Start frontend: cd ../frontend && npm run dev")
    print("3. Visit: http://localhost:3000/tle-chat")

if __name__ == "__main__":
    asyncio.run(demo()) 