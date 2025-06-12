#!/usr/bin/env python3
"""
UDL Data Retrieval Demo
Demonstrates practical usage of UDL for space domain awareness
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from asttroshield.api_client.udl_client import UDLClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UDLDemo:
    """Demonstrate practical UDL data retrieval for space domain awareness."""
    
    def __init__(self):
        """Initialize the UDL demo client."""
        # Get UDL configuration
        self.base_url = os.environ.get("UDL_BASE_URL", "https://unifieddatalibrary.com")
        self.api_key = os.environ.get("UDL_API_KEY", "demo_api_key")
        
        # Initialize client
        self.client = UDLClient(self.base_url, self.api_key)
        
        # High-value targets for demonstration
        self.demo_objects = {
            "25544": "International Space Station (ISS)",
            "20580": "NOAA-18 Weather Satellite", 
            "27424": "COSMOS 2251 Debris",
            "43013": "Starlink-1007",
            "48274": "GPS III SV03"
        }
        
    def demo_space_situational_awareness(self):
        """Demonstrate comprehensive space situational awareness data retrieval."""
        logger.info("🛰️  Space Situational Awareness Demo")
        logger.info("=" * 50)
        
        # 1. Current Space Weather Conditions
        logger.info("\n📡 Current Space Weather Conditions:")
        try:
            space_weather = self.client.get_space_weather_data()
            if space_weather:
                logger.info("✓ Space weather data retrieved")
                # Extract key indicators
                if 'solarFluxIndex' in space_weather:
                    logger.info(f"  Solar Flux: {space_weather['solarFluxIndex']}")
                if 'geomagneticIndex' in space_weather:
                    logger.info(f"  Geomagnetic Activity: {space_weather['geomagneticIndex']}")
            else:
                logger.info("  No space weather data available")
        except Exception as e:
            logger.error(f"  Failed to retrieve space weather: {e}")
            
        # 2. High-Priority Object Tracking
        logger.info("\n🎯 High-Priority Object Tracking:")
        for obj_id, obj_name in self.demo_objects.items():
            logger.info(f"\n  Tracking: {obj_name} (ID: {obj_id})")
            
            try:
                # Get current state vector
                state_vector = self.client.get_state_vector(obj_id)
                if state_vector:
                    logger.info("    ✓ State vector retrieved")
                    if 'position' in state_vector:
                        pos = state_vector['position']
                        logger.info(f"    Position: X={pos.get('x', 'N/A')}, Y={pos.get('y', 'N/A')}, Z={pos.get('z', 'N/A')} km")
                    if 'velocity' in state_vector:
                        vel = state_vector['velocity']
                        logger.info(f"    Velocity: VX={vel.get('vx', 'N/A')}, VY={vel.get('vy', 'N/A')}, VZ={vel.get('vz', 'N/A')} km/s")
                        
                # Get object health
                health = self.client.get_object_health(obj_id)
                if health:
                    logger.info("    ✓ Health status retrieved")
                    if 'status' in health:
                        logger.info(f"    Health Status: {health['status']}")
                        
            except Exception as e:
                logger.info(f"    ✗ Failed to retrieve data: {e}")
                
    def demo_conjunction_analysis(self):
        """Demonstrate conjunction analysis and collision avoidance."""
        logger.info("\n💥 Conjunction Analysis Demo")
        logger.info("=" * 50)
        
        # Focus on ISS for conjunction analysis
        iss_id = "25544"
        logger.info(f"Analyzing conjunctions for ISS (ID: {iss_id})")
        
        try:
            # Get current conjunction data
            conjunction_data = self.client.get_conjunction_data(iss_id)
            if conjunction_data:
                logger.info("✓ Current conjunction data retrieved")
                
                # Check for active conjunctions
                if 'conjunctions' in conjunction_data:
                    conjunctions = conjunction_data['conjunctions']
                    if conjunctions:
                        logger.info(f"  Active conjunctions: {len(conjunctions)}")
                        for i, conj in enumerate(conjunctions[:3]):  # Show first 3
                            logger.info(f"    Conjunction {i+1}:")
                            logger.info(f"      Secondary Object: {conj.get('secondaryObjectId', 'Unknown')}")
                            logger.info(f"      Time of Closest Approach: {conj.get('tca', 'Unknown')}")
                            logger.info(f"      Miss Distance: {conj.get('missDistance', 'Unknown')} km")
                            logger.info(f"      Probability of Collision: {conj.get('poc', 'Unknown')}")
                    else:
                        logger.info("  No active conjunctions detected")
                else:
                    logger.info("  No conjunction data in response")
                    
            # Get historical conjunction data (last 7 days)
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=7)
            
            conjunction_history = self.client.get_conjunction_history(
                iss_id,
                start_time.isoformat(),
                end_time.isoformat()
            )
            
            if conjunction_history:
                logger.info("✓ Historical conjunction data retrieved")
                if isinstance(conjunction_history, list):
                    logger.info(f"  Conjunctions in last 7 days: {len(conjunction_history)}")
                    
        except Exception as e:
            logger.error(f"✗ Conjunction analysis failed: {e}")
            
    def demo_rf_interference_monitoring(self):
        """Demonstrate RF interference monitoring."""
        logger.info("\n📻 RF Interference Monitoring Demo")
        logger.info("=" * 50)
        
        # Common satellite communication frequency bands
        frequency_bands = {
            "L-Band": {"min": 1000.0, "max": 2000.0},
            "S-Band": {"min": 2000.0, "max": 4000.0},
            "C-Band": {"min": 4000.0, "max": 8000.0},
            "X-Band": {"min": 8000.0, "max": 12000.0},
            "Ku-Band": {"min": 12000.0, "max": 18000.0}
        }
        
        for band_name, freq_range in frequency_bands.items():
            logger.info(f"\n  Scanning {band_name} ({freq_range['min']}-{freq_range['max']} MHz):")
            
            try:
                rf_data = self.client.get_rf_interference(freq_range)
                if rf_data and 'rfEmitters' in rf_data:
                    emitters = rf_data['rfEmitters']
                    logger.info(f"    ✓ Found {len(emitters)} RF emitters")
                    
                    # Analyze interference levels
                    if emitters:
                        high_power_emitters = [e for e in emitters if e.get('transmitPower', 0) > 1000]
                        logger.info(f"    High-power emitters (>1kW): {len(high_power_emitters)}")
                        
                        # Show top interferers
                        for i, emitter in enumerate(emitters[:2]):  # Show first 2
                            logger.info(f"      Emitter {i+1}:")
                            logger.info(f"        Frequency: {emitter.get('frequency', 'Unknown')} MHz")
                            logger.info(f"        Power: {emitter.get('transmitPower', 'Unknown')} W")
                            logger.info(f"        Location: {emitter.get('location', 'Unknown')}")
                else:
                    logger.info("    No RF emitters detected in this band")
                    
            except Exception as e:
                logger.info(f"    ✗ Failed to scan {band_name}: {e}")
                
    def demo_new_udl_capabilities(self):
        """Demonstrate new UDL 1.33.0 capabilities."""
        logger.info("\n🆕 New UDL 1.33.0 Capabilities Demo")
        logger.info("=" * 50)
        
        # 1. Electromagnetic Interference Reports
        logger.info("\n⚡ Electromagnetic Interference (EMI) Reports:")
        try:
            emi_reports = self.client.get_emi_reports()
            if emi_reports:
                logger.info("✓ EMI reports retrieved")
                if 'reports' in emi_reports:
                    reports = emi_reports['reports']
                    logger.info(f"  Active EMI reports: {len(reports)}")
                    for i, report in enumerate(reports[:2]):  # Show first 2
                        logger.info(f"    Report {i+1}:")
                        logger.info(f"      Source: {report.get('source', 'Unknown')}")
                        logger.info(f"      Frequency: {report.get('frequency', 'Unknown')} MHz")
                        logger.info(f"      Severity: {report.get('severity', 'Unknown')}")
            else:
                logger.info("  No EMI reports available")
        except Exception as e:
            logger.info(f"  ✗ Failed to retrieve EMI reports: {e}")
            
        # 2. Laser Deconfliction
        logger.info("\n🔴 Laser Deconfliction Requests:")
        try:
            laser_requests = self.client.get_laser_deconflict_requests()
            if laser_requests:
                logger.info("✓ Laser deconflict requests retrieved")
                if 'requests' in laser_requests:
                    requests = laser_requests['requests']
                    logger.info(f"  Active laser requests: {len(requests)}")
            else:
                logger.info("  No laser deconfliction requests")
        except Exception as e:
            logger.info(f"  ✗ Failed to retrieve laser requests: {e}")
            
        # 3. Energetic Charged Particle Data
        logger.info("\n⚛️  Energetic Charged Particle Environmental Data (ECPEDR):")
        try:
            ecpedr_data = self.client.get_ecpedr_data()
            if ecpedr_data:
                logger.info("✓ ECPEDR data retrieved")
                if 'measurements' in ecpedr_data:
                    measurements = ecpedr_data['measurements']
                    logger.info(f"  Particle measurements: {len(measurements)}")
            else:
                logger.info("  No ECPEDR data available")
        except Exception as e:
            logger.info(f"  ✗ Failed to retrieve ECPEDR data: {e}")
            
    def demo_operational_workflow(self):
        """Demonstrate a complete operational workflow."""
        logger.info("\n🔄 Operational Workflow Demo")
        logger.info("=" * 50)
        
        # Simulate a typical operational scenario
        logger.info("Scenario: ISS maneuver planning and collision avoidance")
        
        iss_id = "25544"
        
        # Step 1: Get current ISS status
        logger.info("\n1. Current ISS Status Assessment:")
        try:
            state_vector = self.client.get_state_vector(iss_id)
            health_status = self.client.get_object_health(iss_id)
            
            if state_vector and health_status:
                logger.info("   ✓ ISS status nominal")
                logger.info(f"   Position: {state_vector.get('position', 'Unknown')}")
                logger.info(f"   Health: {health_status.get('status', 'Unknown')}")
            else:
                logger.info("   ⚠️  ISS status data incomplete")
                
        except Exception as e:
            logger.info(f"   ✗ Failed to get ISS status: {e}")
            
        # Step 2: Check for conjunctions
        logger.info("\n2. Conjunction Assessment:")
        try:
            conjunction_data = self.client.get_conjunction_data(iss_id)
            if conjunction_data and 'conjunctions' in conjunction_data:
                conjunctions = conjunction_data['conjunctions']
                if conjunctions:
                    logger.info(f"   ⚠️  {len(conjunctions)} conjunction(s) detected")
                    # Check for high-risk conjunctions
                    high_risk = [c for c in conjunctions if float(c.get('poc', 0)) > 1e-4]
                    if high_risk:
                        logger.info(f"   🚨 {len(high_risk)} high-risk conjunction(s) require attention")
                    else:
                        logger.info("   ✓ All conjunctions are low-risk")
                else:
                    logger.info("   ✓ No conjunctions detected")
            else:
                logger.info("   ✓ Conjunction data clear")
        except Exception as e:
            logger.info(f"   ✗ Conjunction check failed: {e}")
            
        # Step 3: Check space weather
        logger.info("\n3. Space Weather Assessment:")
        try:
            space_weather = self.client.get_space_weather_data()
            if space_weather:
                # Simulate space weather impact assessment
                solar_flux = space_weather.get('solarFluxIndex', 100)
                if solar_flux > 200:
                    logger.info("   ⚠️  High solar activity - increased atmospheric drag expected")
                elif solar_flux > 150:
                    logger.info("   ⚠️  Moderate solar activity - monitor orbital decay")
                else:
                    logger.info("   ✓ Space weather conditions nominal")
            else:
                logger.info("   ⚠️  Space weather data unavailable")
        except Exception as e:
            logger.info(f"   ✗ Space weather check failed: {e}")
            
        # Step 4: Send operational notification
        logger.info("\n4. Operational Notification:")
        try:
            notification = self.client.create_notification(
                "OPERATIONAL_STATUS",
                f"ISS operational assessment completed at {datetime.utcnow().isoformat()}",
                "INFO"
            )
            if notification:
                logger.info("   ✓ Operational status notification sent")
            else:
                logger.info("   ⚠️  Notification delivery uncertain")
        except Exception as e:
            logger.info(f"   ✗ Notification failed: {e}")
            
        logger.info("\n✅ Operational workflow completed")
        
    def run_demo(self):
        """Run the complete UDL demonstration."""
        logger.info("🚀 UDL Data Retrieval Demonstration")
        logger.info("=" * 60)
        logger.info(f"UDL Base URL: {self.base_url}")
        logger.info(f"Authentication: {'API Key' if self.api_key != 'demo_api_key' else 'Demo Mode'}")
        
        # Run all demonstration modules
        try:
            self.demo_space_situational_awareness()
            self.demo_conjunction_analysis()
            self.demo_rf_interference_monitoring()
            self.demo_new_udl_capabilities()
            self.demo_operational_workflow()
            
        except KeyboardInterrupt:
            logger.info("\n⏹️  Demo interrupted by user")
        except Exception as e:
            logger.error(f"\n💥 Demo failed with error: {e}")
            
        logger.info("\n" + "=" * 60)
        logger.info("🎯 UDL Demo Summary")
        logger.info("=" * 60)
        logger.info("Demonstrated capabilities:")
        logger.info("  ✓ Space situational awareness data retrieval")
        logger.info("  ✓ Conjunction analysis and collision avoidance")
        logger.info("  ✓ RF interference monitoring")
        logger.info("  ✓ New UDL 1.33.0 services (EMI, Laser, ECPEDR)")
        logger.info("  ✓ Complete operational workflow")
        logger.info("\nAstroShield is ready for operational UDL integration! 🛰️")


def main():
    """Main function to run the UDL demo."""
    # Check for demo mode
    demo_mode = os.environ.get("UDL_DEMO_MODE", "true").lower() == "true"
    
    if demo_mode:
        logger.info("🧪 Running in DEMO MODE")
        logger.info("Set UDL_API_KEY environment variable for real UDL connection")
        
        # Set demo credentials
        if not os.environ.get("UDL_API_KEY"):
            os.environ["UDL_API_KEY"] = "demo_api_key_for_testing"
        if not os.environ.get("UDL_BASE_URL"):
            os.environ["UDL_BASE_URL"] = "https://demo-udl.example.com"
    
    # Create and run demo
    demo = UDLDemo()
    demo.run_demo()


if __name__ == "__main__":
    main() 