#!/usr/bin/env python
"""UDL Data Analysis Example

This script demonstrates how to use the UDLDataProcessor to analyze
and process data from the Unified Data Layer.

Usage:
    python udl_data_analysis_example.py --object-id <object_id> [--days <days>]
"""

import argparse
import json
import logging
from datetime import datetime

from asttroshield.api_client.udl_client import UDLClient
from asttroshield.udl_integration import USSFDULIntegrator
from asttroshield.udl_data_processor import UDLDataProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_processor(api_key=None, api_url=None):
    """Set up the UDL client, integrator, and processor."""
    # Initialize the UDL client
    udl_client = UDLClient(api_key=api_key, api_url=api_url)
    
    # Initialize the UDL integrator
    udl_integrator = USSFDULIntegrator(udl_client=udl_client)
    
    # Initialize the UDL data processor
    processor = UDLDataProcessor(
        udl_client=udl_client,
        udl_integrator=udl_integrator
    )
    
    return processor, udl_client, udl_integrator

def orbital_data_analysis(processor, object_id, days=7):
    """Perform orbital data analysis for a space object."""
    logger.info(f"Analyzing orbital data for object {object_id} over {days} days")
    
    try:
        # Process orbital data
        orbital_data = processor.process_orbital_data(object_id, days)
        
        # Pretty print the results
        print("\n=== ORBITAL DATA ANALYSIS ===")
        print(f"Object ID: {orbital_data['object_id']}")
        print(f"Time Range: {orbital_data['time_range']['start']} to {orbital_data['time_range']['end']}")
        print(f"Orbit Stability: {orbital_data['orbit_stability']['status']} (confidence: {orbital_data['orbit_stability']['confidence']})")
        print(f"Altitude Profile: {orbital_data['altitude_profile']['min_altitude']} - {orbital_data['altitude_profile']['max_altitude']} km")
        print(f"Orbital Period: {orbital_data['orbital_period']} minutes")
        
        # Print maneuvers if any
        if orbital_data['recent_maneuvers']:
            print("\nRecent Maneuvers:")
            for maneuver in orbital_data['recent_maneuvers']:
                print(f"  - {maneuver['time']}: {maneuver['type']} (ΔV: {maneuver['delta_v']} m/s)")
        else:
            print("\nNo recent maneuvers detected")
            
        return orbital_data
    except Exception as e:
        logger.error(f"Error analyzing orbital data: {str(e)}")
        return None

def conjunction_analysis(processor, object_id, days_ahead=7):
    """Analyze conjunction risks for a space object."""
    logger.info(f"Analyzing conjunctions for object {object_id} over next {days_ahead} days")
    
    try:
        # Analyze conjunction risks
        conjunctions = processor.analyze_conjunction_risk(object_id, days_ahead)
        
        # Pretty print the results
        print("\n=== CONJUNCTION ANALYSIS ===")
        print(f"Object ID: {object_id}")
        print(f"Time Period: Next {days_ahead} days")
        print(f"Total Conjunctions: {len(conjunctions)}")
        
        # Print conjunctions by risk level
        risk_counts = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0}
        
        for conj in conjunctions:
            if "risk_level" in conj:
                risk_counts[conj["risk_level"]] += 1
        
        print("\nRisk Summary:")
        for level, count in risk_counts.items():
            print(f"  - {level}: {count}")
            
        # Print details of high-risk conjunctions
        high_risk = [c for c in conjunctions if c.get("risk_level") in ["CRITICAL", "HIGH"]]
        
        if high_risk:
            print("\nHigh Risk Conjunctions:")
            for i, conj in enumerate(high_risk):
                secondary = conj.get("secondary_object", {})
                secondary_name = secondary.get("name", "Unknown object")
                
                print(f"  {i+1}. {secondary_name}")
                print(f"     - Time: {conj.get('time_of_closest_approach', 'Unknown')}")
                print(f"     - Miss Distance: {conj.get('miss_distance', 0.0):.2f} km")
                print(f"     - Risk Level: {conj.get('risk_level', 'Unknown')}")
                print(f"     - Recommended Actions: {', '.join(conj.get('recommended_actions', []))}")
                
        return conjunctions
    except Exception as e:
        logger.error(f"Error analyzing conjunctions: {str(e)}")
        return None

def space_weather_analysis(processor, object_ids=None):
    """Analyze space weather impacts."""
    logger.info("Analyzing space weather impacts")
    
    try:
        # Analyze space weather
        weather_analysis = processor.analyze_space_weather_impact(object_ids)
        
        # Pretty print the results
        print("\n=== SPACE WEATHER ANALYSIS ===")
        print(f"Timestamp: {weather_analysis.get('timestamp', 'Unknown')}")
        
        # Print overall conditions
        overall = weather_analysis.get("overall_conditions", {})
        print("\nOverall Conditions:")
        print(f"  - Solar Activity: {overall.get('solar_activity_level', 'Unknown')}")
        print(f"  - Geomagnetic Activity: {overall.get('geomagnetic_activity_level', 'Unknown')}")
        print(f"  - Radiation Level: {overall.get('radiation_level', 'Unknown')}")
        print(f"  - Overall Severity: {overall.get('overall_severity', 'Unknown')}")
        
        # Print operational impacts
        impacts = weather_analysis.get("operational_impact", {})
        print("\nOperational Impacts:")
        for impact_type, level in impacts.items():
            formatted_type = impact_type.replace("_", " ").title()
            print(f"  - {formatted_type}: {level}")
            
        # Print object-specific impacts if available
        obj_impacts = weather_analysis.get("object_specific_impacts", {})
        if obj_impacts and object_ids:
            print("\nObject-Specific Impacts:")
            for obj_id, impact in obj_impacts.items():
                print(f"  Object {obj_id}:")
                for impact_type, level in impact.items():
                    if impact_type != "error":
                        formatted_type = impact_type.replace("_", " ").title()
                        print(f"    - {formatted_type}: {level}")
                
        return weather_analysis
    except Exception as e:
        logger.error(f"Error analyzing space weather: {str(e)}")
        return None

def anomaly_detection(processor, object_id, days=30):
    """Detect anomalies for a space object."""
    logger.info(f"Detecting anomalies for object {object_id} over {days} days")
    
    try:
        # Detect anomalies
        anomalies = processor.detect_anomalies(object_id, days)
        
        # Pretty print the results
        print("\n=== ANOMALY DETECTION ===")
        print(f"Object ID: {anomalies.get('object_id', 'Unknown')}")
        print(f"Time Range: {anomalies.get('time_range', {}).get('start', 'Unknown')} to {anomalies.get('time_range', {}).get('end', 'Unknown')}")
        print(f"Anomalies Detected: {anomalies.get('anomalies_detected', 0)}")
        
        # Print anomalies if any
        if anomalies.get('anomalies', []):
            print("\nDetected Anomalies:")
            for i, anomaly in enumerate(anomalies.get('anomalies', [])):
                print(f"  {i+1}. {anomaly.get('type', 'Unknown Type')}")
                print(f"     - Severity: {anomaly.get('severity', 'Unknown')}")
                print(f"     - Detection Time: {anomaly.get('detection_time', 'Unknown')}")
                print(f"     - Details: {anomaly.get('details', 'No details available')}")
        else:
            print("\nNo anomalies detected")
            
        # Print recommendations
        if anomalies.get('recommendations', []):
            print("\nRecommendations:")
            for rec in anomalies.get('recommendations', []):
                print(f"  - {rec}")
                
        return anomalies
    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
        return None

def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="UDL Data Analysis Example")
    parser.add_argument("--object-id", required=True, help="ID of the space object to analyze")
    parser.add_argument("--days", type=int, default=7, help="Number of days of historical data to analyze")
    parser.add_argument("--days-ahead", type=int, default=7, help="Number of days ahead to look for conjunctions")
    parser.add_argument("--api-key", help="UDL API key (optional)")
    parser.add_argument("--api-url", help="UDL API URL (optional)")
    parser.add_argument("--output-file", help="Output file for results (optional)")
    parser.add_argument("--analyze-weather", action="store_true", help="Analyze space weather impacts")
    parser.add_argument("--detect-anomalies", action="store_true", help="Detect anomalies for the object")
    
    args = parser.parse_args()
    
    # Setup processor
    processor, _, _ = setup_processor(api_key=args.api_key, api_url=args.api_url)
    
    # Collect results
    results = {}
    
    # Orbital data analysis
    orbital_data = orbital_data_analysis(processor, args.object_id, args.days)
    if orbital_data:
        results["orbital_data"] = orbital_data
    
    # Conjunction analysis
    conjunctions = conjunction_analysis(processor, args.object_id, args.days_ahead)
    if conjunctions:
        results["conjunctions"] = conjunctions
    
    # Space weather analysis (optional)
    if args.analyze_weather:
        weather = space_weather_analysis(processor, [args.object_id])
        if weather:
            results["space_weather"] = weather
    
    # Anomaly detection (optional)
    if args.detect_anomalies:
        anomalies = anomaly_detection(processor, args.object_id, args.days)
        if anomalies:
            results["anomalies"] = anomalies
    
    # Print processing metrics
    metrics = processor.get_processing_metrics()
    print("\n=== PROCESSING METRICS ===")
    for metric, value in metrics.items():
        formatted_metric = metric.replace("_", " ").title()
        print(f"{formatted_metric}: {value}")
    
    # Save results to file if requested
    if args.output_file:
        try:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output_file}")
        except Exception as e:
            logger.error(f"Error saving results to file: {str(e)}")

if __name__ == "__main__":
    main() 