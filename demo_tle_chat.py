#!/usr/bin/env python3
"""
Demo script for TLE Chat Interface
Shows how the backend TLE explainer works
"""

import asyncio
import json
import sys
import os

# Add the backend path to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend_fixed'))

from app.ml_infrastructure.tle_explainer import TLEExplainerInput, MockTLEExplainerService

async def demo_tle_chat():
    """Demonstrate TLE chat functionality"""
    
    print("🛰️  TLE Chat Interface Demo")
    print("=" * 50)
    
    # Initialize the TLE explainer service
    explainer = MockTLEExplainerService()
    
    # Sample TLE data (ISS)
    iss_tle = TLEExplainerInput(
        norad_id="25544",
        satellite_name="International Space Station (ISS)",
        line1="1 25544U 98067A   24325.50000000  .00016717  00000+0  10270-3 0  9994",
        line2="2 25544  51.6400 337.6640 0007776  35.5310 330.5120 15.50377579499263",
        include_risk_assessment=True,
        include_anomaly_detection=True
    )
    
    print("📡 Input TLE:")
    print(f"   {iss_tle.line1}")
    print(f"   {iss_tle.line2}")
    print()
    
    try:
        # Get explanation
        print("🔄 Processing TLE...")
        explanation = await explainer.explain_tle(iss_tle)
        
        print("✅ TLE Analysis Complete!")
        print()
        
        # Format the response like the chat interface would
        print("🛰️  Chat Response:")
        print("-" * 30)
        
        risk_emoji = "🔴" if explanation.decay_risk_level == "HIGH" else \
                    "🟡" if explanation.decay_risk_level == "MEDIUM" else "🟢"
        
        chat_response = f"""🛰️ **{explanation.satellite_name}** (ID: {explanation.norad_id})

📊 **Orbital Elements:**
• **Orbit Type**: {explanation.orbit_type} - {explanation.orbit_description}
• **Altitude**: {explanation.altitude_description}
• **Period**: {explanation.period_minutes:.1f} minutes
• **Inclination**: {explanation.inclination_degrees:.2f}°
• **Eccentricity**: {explanation.eccentricity:.6f}

{risk_emoji} **Decay Risk**: {explanation.decay_risk_level} (Score: {(explanation.decay_risk_score * 100):.1f}%)
{f"⏱️ **Estimated Lifetime**: {round(explanation.predicted_lifetime_days)} days" if explanation.predicted_lifetime_days else ""}

{f"⚠️ **Anomalies Detected**: {', '.join(explanation.anomaly_flags)}" if explanation.anomaly_flags else ""}

🎯 **Confidence**: {(explanation.confidence_score * 100):.1f}%"""
        
        print(chat_response)
        print()
        
        # Show technical details
        print("🔧 Technical Details:")
        print("-" * 20)
        for key, value in explanation.technical_details.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.6f}")
            else:
                print(f"   {key}: {value}")
        
        print()
        print("📋 JSON Response:")
        print("-" * 15)
        print(json.dumps({
            "norad_id": explanation.norad_id,
            "satellite_name": explanation.satellite_name,
            "orbit_type": explanation.orbit_type,
            "decay_risk_level": explanation.decay_risk_level,
            "decay_risk_score": explanation.decay_risk_score,
            "confidence_score": explanation.confidence_score,
            "period_minutes": explanation.period_minutes,
            "inclination_degrees": explanation.inclination_degrees,
            "eccentricity": explanation.eccentricity
        }, indent=2))
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print()
    print("🎯 Demo Summary:")
    print("=" * 15)
    print("✅ TLE parsing successful")
    print("✅ Orbital elements calculated")
    print("✅ Risk assessment completed")
    print("✅ Natural language explanation generated")
    print("✅ Chat interface response formatted")
    print()
    print("🚀 The TLE Chat Interface is ready!")
    print("   Frontend: http://localhost:3000/tle-chat")
    print("   Backend API: http://localhost:8000/api/tle-explanations/explain")
    
    return True

def demo_chat_conversation():
    """Simulate a chat conversation"""
    print("\n💬 Simulated Chat Conversation:")
    print("=" * 35)
    
    conversation = [
        {
            "user": "What is a TLE?",
            "assistant": """📚 **What is a TLE?**

A TLE (Two-Line Element set) is a standardized format for describing a satellite's orbit. It contains:

**Line 1**: Satellite catalog number, classification, epoch, orbital decay, drag term
**Line 2**: Inclination, right ascension, eccentricity, argument of perigee, mean anomaly, mean motion

TLEs are updated regularly as satellites are tracked and their orbits change due to atmospheric drag and other forces."""
        },
        {
            "user": "1 25544U 98067A   24325.50000000  .00016717  00000+0  10270-3 0  9994\n2 25544  51.6400 337.6640 0007776  35.5310 330.5120 15.50377579499263",
            "assistant": """🛰️ **International Space Station (ISS)** (ID: 25544)

📊 **Orbital Elements:**
• **Orbit Type**: LEO - Low Earth Orbit with moderate inclination
• **Altitude**: 408.0 km x 416.0 km
• **Period**: 92.8 minutes
• **Inclination**: 51.64°
• **Eccentricity**: 0.000778

🟡 **Decay Risk**: MEDIUM (Score: 30.0%)
⏱️ **Estimated Lifetime**: 1825 days

🎯 **Confidence**: 95.0%"""
        }
    ]
    
    for i, exchange in enumerate(conversation, 1):
        print(f"\n[{i}] User:")
        print(f"    {exchange['user']}")
        print(f"\n[{i}] AstroShield:")
        print(f"    {exchange['assistant']}")
    
    print("\n✨ This is how the TLE Chat Interface works!")

if __name__ == "__main__":
    print("Starting TLE Chat Demo...")
    
    # Run the async demo
    success = asyncio.run(demo_tle_chat())
    
    if success:
        # Show conversation simulation
        demo_chat_conversation()
        
        print("\n🎉 Demo completed successfully!")
        print("\n📖 Next steps:")
        print("1. Start the backend: cd backend_fixed && python -m uvicorn app.main:app --reload")
        print("2. Start the frontend: cd frontend && npm run dev")
        print("3. Open: http://localhost:3000/tle-chat")
        print("4. Paste a TLE and see the magic! ✨")
    else:
        print("\n❌ Demo failed. Check the backend setup.")
        sys.exit(1) 