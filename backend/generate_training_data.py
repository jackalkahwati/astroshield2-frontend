"""
UDL Data Collection and Training Example Generation

This script collects data from the Unified Data Library (UDL) 
and creates training examples for ML model fine-tuning.
"""
import os
import json
import time
import logging
from pathlib import Path
import requests
from datetime import datetime

# Import configuration
from config import config
from udl_auth import setup_auth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def get_latest_offset(session, topic):
    """Get the latest offset for a topic using Secure Messaging API"""
    try:
        url = f"{config['udl_base_url']}/{config['udl_api_version']}/getLatestOffset/{topic}"
        response = session.get(url)
        response.raise_for_status()
        
        # The response might be the offset directly or in a JSON structure
        try:
            offset_data = response.json()
            if isinstance(offset_data, dict):
                return offset_data.get("offset") or offset_data.get("latestOffset")
            elif isinstance(offset_data, int):
                return offset_data
        except:
            # If not JSON, try to parse as plain text integer
            return int(response.text.strip())
        
        logger.error(f"Unexpected response format from getLatestOffset for {topic}")
        return None
    except requests.RequestException as e:
        logger.error(f"Error getting latest offset for {topic}: {str(e)}")
        return None

def get_messages(session, topic, offset, filters=None):
    """Get messages from a topic using Secure Messaging API"""
    try:
        url = f"{config['udl_base_url']}/{config['udl_api_version']}/getMessages/{topic}/{offset}"
        params = {}
        
        # Add any additional filters
        if filters:
            for key, value in filters.items():
                params[key] = value
        
        # Only add params if we have any
        if params:
            response = session.get(url, params=params)
        else:
            response = session.get(url)
            
        response.raise_for_status()
        
        # Parse the response - format might be different than our original assumption
        data = response.json()
        
        # Handle different possible response formats
        if isinstance(data, list):
            # If direct list of messages
            messages = data
            next_offset = offset + len(messages)
        elif isinstance(data, dict):
            # If structured with metadata
            messages = data.get("messages", []) or data.get("data", [])
            next_offset = data.get("nextOffset", offset + len(messages))
        else:
            logger.error(f"Unexpected response format from getMessages for {topic}")
            return [], offset
        
        return messages, next_offset
    except requests.RequestException as e:
        logger.error(f"Error getting messages for {topic}: {str(e)}")
        return [], offset

def collect_topic_data(session, topic, limit=None):
    """Collect data for a specific topic"""
    if limit is None:
        limit = config["topic_limits"].get(topic, 100)
    
    logger.info(f"Collecting data for topic: {topic} (limit: {limit} records)")
    
    latest_offset = get_latest_offset(session, topic)
    if latest_offset is None:
        logger.warning(f"Could not get latest offset for topic {topic}")
        return []
    
    logger.info(f"Latest offset for {topic}: {latest_offset}")
    
    # Start from a reasonable offset to get recent data
    current_offset = max(0, latest_offset - 1000)  
    collected_data = []
    
    while len(collected_data) < limit:
        messages, next_offset = get_messages(session, topic, current_offset)
        
        if messages:
            collected_data.extend(messages)
            logger.info(f"  Collected {len(messages)} records from {topic}")
            
            # If we've reached the end of the topic, break
            if next_offset == current_offset:
                break
            
            current_offset = next_offset
        else:
            logger.warning(f"  No messages retrieved for {topic} at offset {current_offset}")
            break
        
        # Add a small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Limit to requested sample size
    return collected_data[:limit]

def collect_all_topics(session):
    """Collect data from all configured topics"""
    all_data = {}
    
    # Create output directories
    raw_data_dir = os.path.join(config["output_dir"], config["raw_data_dir"])
    os.makedirs(raw_data_dir, exist_ok=True)
    
    # Collect data for each topic
    for topic in config["topics"]:
        topic_data = collect_topic_data(session, topic)
        all_data[topic] = topic_data
        
        # Save raw data
        with open(os.path.join(raw_data_dir, f"{topic}_data.json"), 'w') as f:
            json.dump(topic_data, f, indent=2)
        
        logger.info(f"Saved {len(topic_data)} records for topic {topic}")
    
    return all_data

def generate_conjunction_examples(conjunction_data):
    """Generate training examples for conjunction events"""
    examples = []
    
    for conjunction in conjunction_data:
        if not conjunction:
            continue
            
        # Extract key information
        primary_sat = conjunction.get("primary", {}).get("satNo", "unknown")
        secondary_sat = conjunction.get("secondary", {}).get("satNo", "unknown")
        tca = conjunction.get("tca", "unknown")
        pc = conjunction.get("pc", "unknown")
        miss_distance = conjunction.get("missDistance", "unknown")
        
        # Generate question
        question = f"What's the status of the conjunction between satellites {primary_sat} and {secondary_sat}?"
        
        # Generate answer
        answer = f"UDL data shows a conjunction between satellites {primary_sat} and {secondary_sat} with a time of closest approach (TCA) at {tca}. "
        answer += f"The probability of collision is {pc}, with a miss distance of {miss_distance} meters."
        
        # Add additional information if available
        if conjunction.get("screeningEntity"):
            answer += f" This conjunction was reported by {conjunction.get('screeningEntity')}."
        
        examples.append({
            "text_input": question,
            "output": answer
        })
    
    return examples

def generate_maneuver_examples(maneuver_data, state_vector_data=[]):
    """Generate training examples for maneuvers"""
    examples = []
    
    for maneuver in maneuver_data:
        if not maneuver:
            continue
            
        # Extract key information
        sat_no = maneuver.get("satNo", "unknown")
        maneuver_time = maneuver.get("maneuverTime", "unknown")
        delta_v = maneuver.get("deltaV", "unknown")
        maneuver_type = maneuver.get("maneuverType", "unknown")
        
        # Generate question
        question = f"Our analysts detected a possible maneuver by satellite {sat_no}. Can you provide details about this maneuver?"
        
        # Generate detailed answer
        answer = f"UDL data confirms that satellite {sat_no} performed a {maneuver_type} maneuver on {maneuver_time}. The maneuver had a delta-V of approximately {delta_v} m/s."
        
        # Add additional details if available
        if maneuver.get("purpose"):
            answer += f" The assessed purpose of this maneuver was {maneuver.get('purpose')}."
        if maneuver.get("detectionConfidence"):
            answer += f" This maneuver was detected with {maneuver.get('detectionConfidence')} confidence."
            
        examples.append({
            "text_input": question,
            "output": answer
        })
    
    return examples

def generate_observation_examples(eo_data, radar_data, rf_data):
    """Generate training examples that combine different observation types"""
    examples = []
    
    # Group observations by satellite number
    satellites = {}
    
    # Process EO observations
    for obs in eo_data:
        if not obs:
            continue
        sat_no = obs.get("satNo")
        if sat_no:
            if sat_no not in satellites:
                satellites[sat_no] = {"eo": [], "radar": [], "rf": []}
            satellites[sat_no]["eo"].append(obs)
    
    # Process radar observations
    for obs in radar_data:
        if not obs:
            continue
        sat_no = obs.get("satNo")
        if sat_no:
            if sat_no not in satellites:
                satellites[sat_no] = {"eo": [], "radar": [], "rf": []}
            satellites[sat_no]["radar"].append(obs)
    
    # Process RF observations
    for obs in rf_data:
        if not obs:
            continue
        sat_no = obs.get("satNo")
        if sat_no:
            if sat_no not in satellites:
                satellites[sat_no] = {"eo": [], "radar": [], "rf": []}
            satellites[sat_no]["rf"].append(obs)
    
    # Generate examples for satellites with multiple observation types
    for sat_no, data in satellites.items():
        if len(data["eo"]) > 0 or len(data["radar"]) > 0 or len(data["rf"]) > 0:
            question = f"Can you provide a comprehensive analysis of all sensor observations for satellite {sat_no} from the last week?"
            
            answer = f"Analysis of UDL data for satellite {sat_no} shows the following observation patterns:\n\n"
            
            if data["eo"]:
                answer += f"Electro-Optical Observations: {len(data['eo'])} observations recorded. "
                latest_eo = max(data["eo"], key=lambda x: x.get("obTime", ""))
                answer += f"The most recent EO observation was at {latest_eo.get('obTime')}. "
                if latest_eo.get("magnitude"):
                    answer += f"The satellite had a visual magnitude of {latest_eo.get('magnitude')}. "
            
            if data["radar"]:
                answer += f"\n\nRadar Observations: {len(data['radar'])} observations recorded. "
                latest_radar = max(data["radar"], key=lambda x: x.get("obTime", ""))
                answer += f"The most recent radar observation was at {latest_radar.get('obTime')}. "
                if latest_radar.get("rcs"):
                    answer += f"The radar cross section was measured at {latest_radar.get('rcs')} square meters. "
            
            if data["rf"]:
                answer += f"\n\nRF Observations: {len(data['rf'])} observations recorded. "
                latest_rf = max(data["rf"], key=lambda x: x.get("obTime", ""))
                answer += f"The most recent RF observation was at {latest_rf.get('obTime')}. "
                if latest_rf.get("frequency"):
                    answer += f"The detected signal frequency was {latest_rf.get('frequency')} MHz. "
            
            examples.append({
                "text_input": question,
                "output": answer
            })
    
    return examples

def generate_aircraft_examples(aircraft_data):
    """Generate training examples for aircraft data"""
    examples = []
    
    for aircraft in aircraft_data:
        if not aircraft:
            continue
            
        # Extract key information
        tail_number = aircraft.get("tailNumber", "unknown")
        aircraft_mds = aircraft.get("aircraftMDS", "unknown")
        category = aircraft.get("category", "unknown")
        owner = aircraft.get("owner", "unknown")
        
        # Generate questions about specific aircraft
        question1 = f"What type of aircraft is tail number {tail_number}?"
        answer1 = f"According to UDL data, tail number {tail_number} is a {aircraft_mds} aircraft."
        if category:
            answer1 += f" It falls under the {category} category."
        if owner:
            answer1 += f" It is operated by {owner}."
        
        examples.append({
            "text_input": question1,
            "output": answer1
        })
        
        # Generate questions about aircraft capabilities
        if aircraft.get("cruiseSpeed") or aircraft.get("maxSpeed"):
            question2 = f"What are the speed capabilities of the {aircraft_mds} with tail number {tail_number}?"
            answer2 = f"The {aircraft_mds} with tail number {tail_number} has the following performance characteristics:"
            
            if aircraft.get("cruiseSpeed"):
                answer2 += f" Cruise speed: {aircraft.get('cruiseSpeed')} km/h."
            
            if aircraft.get("maxSpeed"):
                answer2 += f" Maximum speed: {aircraft.get('maxSpeed')} km/h."
                
            examples.append({
                "text_input": question2,
                "output": answer2
            })
    
    return examples

def generate_all_training_examples(all_data):
    """Generate all types of training examples"""
    all_examples = []
    
    # Generate aircraft examples
    if "aircraft" in all_data and all_data["aircraft"]:
        aircraft_examples = generate_aircraft_examples(all_data["aircraft"])
        all_examples.extend(aircraft_examples)
        logger.info(f"Generated {len(aircraft_examples)} aircraft examples")
    
    # Generate conjunction examples
    if "conjunction" in all_data and all_data["conjunction"]:
        conjunction_examples = generate_conjunction_examples(all_data["conjunction"])
        all_examples.extend(conjunction_examples)
        logger.info(f"Generated {len(conjunction_examples)} conjunction examples")
    
    # Generate maneuver examples
    if "maneuver" in all_data and all_data["maneuver"]:
        maneuver_examples = generate_maneuver_examples(
            all_data["maneuver"], 
            all_data.get("statevector", [])
        )
        all_examples.extend(maneuver_examples)
        logger.info(f"Generated {len(maneuver_examples)} maneuver examples")
    
    # Generate observation examples
    if all_data.get("eoobservation") or all_data.get("radarobservation") or all_data.get("rfobservation"):
        observation_examples = generate_observation_examples(
            all_data.get("eoobservation", []),
            all_data.get("radarobservation", []),
            all_data.get("rfobservation", [])
        )
        all_examples.extend(observation_examples)
        logger.info(f"Generated {len(observation_examples)} observation examples")
    
    # Save all examples
    os.makedirs(config["output_dir"], exist_ok=True)
    with open(os.path.join(config["output_dir"], "training_examples.json"), 'w') as f:
        json.dump(all_examples, f, indent=2)
    
    logger.info(f"Generated a total of {len(all_examples)} training examples")
    return all_examples

def main():
    """Main execution function"""
    logger.info("Starting UDL data collection for training dataset generation")
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    try:
        # Set up authentication
        session = setup_auth()
        
        # Collect data from all topics
        all_data = collect_all_topics(session)
        
        # Generate training examples if configured
        if config["generate_examples"]:
            training_examples = generate_all_training_examples(all_data)
            
            # Format for fine-tuning
            formatted_examples = []
            for example in training_examples:
                formatted_examples.append({
                    "text_input": example["text_input"],
                    "output": example["output"]
                })
            
            # Save formatted examples for fine-tuning
            with open(os.path.join(config["output_dir"], "gemini_training_data.json"), 'w') as f:
                json.dump(formatted_examples, f, indent=2)
            
            logger.info(f"Successfully generated {len(formatted_examples)} examples for fine-tuning")
            logger.info(f"Training data saved to {os.path.join(config['output_dir'], 'gemini_training_data.json')}")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()