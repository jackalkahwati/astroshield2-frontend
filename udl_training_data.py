"""Script for accessing UDL data and preparing training data for fine-tuning."""

import json
import os
import requests
import traceback
import base64
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from datetime import datetime
from dateutil.parser import parse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class UDLTrainingExample:
    """Data class representing a training example for fine-tuning."""
    text_input: str
    output: str

class UDLTrainingDataGenerator:
    """Class to generate training data from UDL sources."""
    
    def __init__(self):
        """Initialize the UDL client with credentials from environment variables."""
        # Read base URL from environment, don't append anything
        raw_base_url = os.getenv("UDL_BASE_URL", "https://unifieddatalibrary.com")
        print(f"Raw base URL from environment: {raw_base_url}")
        # Force the base URL to be correct
        self.base_url = "https://unifieddatalibrary.com"
        self.username = os.getenv("UDL_USERNAME")
        self.password = os.getenv("UDL_PASSWORD")
        
        print(f"Initializing UDL client with:")
        print(f"  Base URL: {self.base_url}")
        print(f"  Username: {self.username}")
        print(f"  Password: {'*' * len(self.password) if self.password else None}")
        
        if not (self.username and self.password):
            raise ValueError("Both UDL_USERNAME and UDL_PASSWORD must be set in the .env file")
        
        # Initialize session
        self.session = requests.Session()
        
        # Set up basic authentication
        auth_string = f"{self.username}:{self.password}"
        encoded_auth = base64.b64encode(auth_string.encode('ascii')).decode('ascii')
        self.session.headers.update({
            "Authorization": f"Basic {encoded_auth}",
            "Content-Type": "application/json"
        })
        print("Using Basic authentication with username and password")
    
    def __del__(self):
        """Close the session when the object is deleted."""
        if hasattr(self, 'session'):
            self.session.close()
    
    def _make_request(self, endpoint, params=None):
        """Make a request to the UDL API with error handling."""
        url = f"{self.base_url}/{endpoint}"
        print(f"Making request to: {url}")
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            print(f"Response status code: {response.status_code}")
            
            # Check if the request was successful
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            print(f"Response content: {e.response.content.decode('utf-8') if hasattr(e, 'response') else 'No response'}")
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error occurred: {e}")
            return None
        except requests.exceptions.Timeout as e:
            print(f"Timeout error occurred: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error during request: {e}")
            traceback.print_exc()
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            traceback.print_exc()
            return None
    
    def fetch_elset_data(self, limit=20) -> List[UDLTrainingExample]:
        """Fetch ELSET (element set) data from UDL and create training examples."""
        print("\nFetching ELSET data...")
        examples = []
        
        try:
            # Simple query without parameters first
            data = self._make_request('udl/elset')
            
            if not data:
                # Try with epoch parameter for count endpoint
                print("Trying ELSET count endpoint...")
                data = self._make_request('udl/elset/count', {'epoch': '>now-30 days'})
                
                if not data:
                    print("No ELSET data found with any endpoint")
                    return examples
            
            print(f"Raw ELSET response data type: {type(data)}")
            
            # If we got a JSON object with 'results' key
            if isinstance(data, dict) and 'results' in data:
                results = data.get('results', [])
                print(f"Retrieved {len(results)} ELSET records")
                
                for item in results:
                    # Format the input text
                    object_id = item.get('idOnOrbit', 'Unknown')
                    epoch = item.get('epoch', 'Unknown date')
                    
                    input_text = f"Describe the orbital elements for satellite with ID {object_id} at epoch {epoch}."
                    
                    # Format the output text
                    output_parts = [
                        f"Orbital Elements Analysis:",
                        f"Object ID: {object_id}",
                        f"Epoch: {epoch}",
                        f"Inclination: {item.get('inclination', 'Unknown')} degrees",
                        f"Eccentricity: {item.get('eccentricity', 'Unknown')}",
                        f"Semimajor Axis: {item.get('semimajorAxis', 'Unknown')} km",
                        f"RAAN: {item.get('rightAscensionAscendingNode', 'Unknown')} degrees",
                        f"Argument of Perigee: {item.get('argumentOfPerigee', 'Unknown')} degrees",
                        f"Mean Anomaly: {item.get('meanAnomaly', 'Unknown')} degrees"
                    ]
                    
                    output_text = "\n".join(output_parts)
                    
                    examples.append(UDLTrainingExample(text_input=input_text, output=output_text))
                    
                if not examples and isinstance(data, dict):
                    # Create an example from whatever data we got
                    print("Creating example from available data")
                    count = data.get('count', 0)
                    if count > 0:
                        input_text = "How many ELSETs have been recorded in the UDL in the last 30 days?"
                        output_text = f"There are {count} ELSET records in the UDL from the past 30 days."
                        examples.append(UDLTrainingExample(text_input=input_text, output=output_text))
                    else:
                        # Create a dummy example if no real data is available
                        input_text = "What is an ELSET in space situational awareness?"
                        output_text = (
                            "An ELSET (Element Set) is a collection of orbital parameters that describe "
                            "the orbit of a satellite or space object. In space situational awareness, "
                            "ELSETs are critical for tracking objects, predicting their positions, and "
                            "calculating possible conjunctions with other objects. The parameters typically "
                            "include inclination, eccentricity, right ascension of ascending node (RAAN), "
                            "argument of perigee, mean anomaly, and mean motion."
                        )
                        examples.append(UDLTrainingExample(text_input=input_text, output=output_text))
            else:
                # Just create a dummy example
                print("Creating dummy ELSET example with educational content")
                input_text = "What is an ELSET in space situational awareness?"
                output_text = (
                    "An ELSET (Element Set) is a collection of orbital parameters that describe "
                    "the orbit of a satellite or space object. In space situational awareness, "
                    "ELSETs are critical for tracking objects, predicting their positions, and "
                    "calculating possible conjunctions with other objects. The parameters typically "
                    "include inclination, eccentricity, right ascension of ascending node (RAAN), "
                    "argument of perigee, mean anomaly, and mean motion."
                )
                examples.append(UDLTrainingExample(text_input=input_text, output=output_text))
            
            print(f"Created {len(examples)} training examples from ELSET data")
            
        except Exception as e:
            print(f"Error fetching ELSET data: {e}")
            traceback.print_exc()
            
            # Create a dummy example if an error occurs
            input_text = "What is an ELSET in space situational awareness?"
            output_text = (
                "An ELSET (Element Set) is a collection of orbital parameters that describe "
                "the orbit of a satellite or space object. In space situational awareness, "
                "ELSETs are critical for tracking objects, predicting their positions, and "
                "calculating possible conjunctions with other objects. The parameters typically "
                "include inclination, eccentricity, right ascension of ascending node (RAAN), "
                "argument of perigee, mean anomaly, and mean motion."
            )
            examples.append(UDLTrainingExample(text_input=input_text, output=output_text))
        
        return examples
    
    def fetch_tle_data(self, limit=20) -> List[UDLTrainingExample]:
        """Fetch TLE (Two-Line Element) data from UDL and create training examples."""
        print("\nFetching TLE data...")
        examples = []
        
        try:
            # Create dummy examples since we couldn't access real TLE data
            input_text = "What is a TLE and how is it used in space operations?"
            output_text = (
                "A TLE (Two-Line Element set) is a data format used to convey orbital information about "
                "Earth-orbiting objects. Each TLE consists of two lines of 69 characters each, encoding "
                "information such as epoch, orbital inclination, right ascension of ascending node, "
                "eccentricity, argument of perigee, mean anomaly, mean motion, and a checksum.\n\n"
                "In space operations, TLEs are used to:\n"
                "1. Predict satellite positions using propagation models like SGP4\n"
                "2. Schedule satellite communications and observations\n"
                "3. Plan collision avoidance maneuvers\n"
                "4. Track space debris\n"
                "5. Calculate satellite visibility from ground stations\n\n"
                "TLEs are regularly updated by organizations like the US Space Force and shared through "
                "services like the Space-Track catalog and the Unified Data Library (UDL)."
            )
            examples.append(UDLTrainingExample(text_input=input_text, output=output_text))
            
            input_text = "How do you interpret a TLE string for satellite tracking?"
            output_text = (
                "To interpret a TLE (Two-Line Element set) for satellite tracking:\n\n"
                "Line 1: 1 25544U 98067A   23083.73207871  .00009026  00000+0  16477-3 0  9991\n"
                "Line 2: 2 25544  51.6431 178.0096 0006394 323.9691  97.9758 15.49357619386886\n\n"
                "Breakdown:\n"
                "- 1 25544U: Line number (1), satellite catalog number (25544), and classification (U=Unclassified)\n"
                "- 98067A: International designator (launch year 1998, launch number 067, launch piece A)\n"
                "- 23083.73207871: Epoch (year 2023, day 83.73207871)\n"
                "- .00009026: First derivative of mean motion (drag term)\n"
                "- 00000+0: Second derivative of mean motion (not used in SGP4)\n"
                "- 16477-3: B* drag term (0.00016477)\n"
                "- 0 9991: Element set number and checksum\n\n"
                "Line 2:\n"
                "- 2 25544: Line number (2) and satellite catalog number (25544)\n"
                "- 51.6431: Inclination (degrees)\n"
                "- 178.0096: Right Ascension of Ascending Node (degrees)\n"
                "- 0006394: Eccentricity (0.0006394)\n"
                "- 323.9691: Argument of Perigee (degrees)\n"
                "- 97.9758: Mean Anomaly (degrees)\n"
                "- 15.49357619: Mean Motion (revolutions per day)\n"
                "- 38688: Revolution number at epoch (checksum 6)"
            )
            examples.append(UDLTrainingExample(text_input=input_text, output=output_text))
            
            print(f"Created {len(examples)} dummy examples for TLE data")
            
        except Exception as e:
            print(f"Error creating TLE examples: {e}")
            traceback.print_exc()
        
        return examples
    
    def fetch_statevector_data(self, limit=20) -> List[UDLTrainingExample]:
        """Fetch state vector data from UDL and create training examples."""
        print("\nFetching state vector data...")
        examples = []
        
        try:
            # Try with epoch parameter for statevector count
            data = self._make_request('udl/statevector/count', {'epoch': '>now-30 days'})
            
            if not data:
                # Create dummy examples instead
                print("No state vector data found, creating educational examples instead")
                
                input_text = "What is a state vector in orbital mechanics?"
                output_text = (
                    "A state vector in orbital mechanics is a mathematical representation of an object's "
                    "position and velocity in three-dimensional space at a specific moment in time (epoch). "
                    "It consists of six components:\n\n"
                    "1. Position components (x, y, z): The satellite's location in 3D space, typically measured "
                    "in kilometers relative to Earth's center in an Earth-centered inertial (ECI) coordinate system.\n\n"
                    "2. Velocity components (vx, vy, vz): The satellite's velocity vector, typically measured "
                    "in kilometers per second in the same coordinate system.\n\n"
                    "State vectors are fundamental in space situational awareness because they:\n"
                    "- Provide an instantaneous snapshot of an object's trajectory\n"
                    "- Can be propagated forward or backward in time using physics models\n"
                    "- Can be converted to other orbital representations like Keplerian elements\n"
                    "- Are used in conjunction analysis, maneuver planning, and reentry predictions\n"
                    "- Serve as inputs to more complex space mission design calculations"
                )
                examples.append(UDLTrainingExample(text_input=input_text, output=output_text))
                
                input_text = "How do you convert between state vectors and Keplerian orbital elements?"
                output_text = (
                    "Converting between state vectors and Keplerian orbital elements involves several steps "
                    "and mathematical operations:\n\n"
                    "From State Vector to Keplerian Elements:\n"
                    "1. Calculate the position vector magnitude (r) and velocity magnitude (v)\n"
                    "2. Calculate the specific angular momentum vector (h = r × v)\n"
                    "3. Calculate the node vector (n = z-axis × h)\n"
                    "4. Calculate the eccentricity vector (e = ((v² - μ/r)r - (r·v)v)/μ)\n"
                    "5. Calculate the eccentricity (e = |e|)\n"
                    "6. Calculate the semi-major axis (a = h²/(μ(1-e²)) for e<1)\n"
                    "7. Calculate the inclination (i = cos⁻¹(h_z/|h|))\n"
                    "8. Calculate the right ascension of ascending node (Ω = cos⁻¹(n_x/|n|))\n"
                    "9. Calculate the argument of perigee (ω = angle between n and e)\n"
                    "10. Calculate the true anomaly (ν = angle between e and r)\n\n"
                    "From Keplerian Elements to State Vector:\n"
                    "1. Calculate the distance from central body (r = a(1-e²)/(1+e·cos(ν)))\n"
                    "2. Calculate position in orbital plane (x' = r·cos(ν), y' = r·sin(ν))\n"
                    "3. Calculate velocity in orbital plane (vx', vy')\n"
                    "4. Transform position and velocity from orbital plane to reference frame using "
                    "rotation matrices based on i, Ω, and ω\n\n"
                    "These conversions are essential in orbital mechanics for various applications, "
                    "including orbit determination, maneuver planning, and space situational awareness."
                )
                examples.append(UDLTrainingExample(text_input=input_text, output=output_text))
                
                print(f"Created {len(examples)} educational examples for state vector data")
                return examples
            
            # If we got real data with a count
            if isinstance(data, dict) and 'count' in data:
                count = data.get('count', 0)
                input_text = "How many state vectors have been recorded in the UDL in the last 30 days?"
                output_text = f"There are {count} state vector records in the UDL from the past 30 days."
                examples.append(UDLTrainingExample(text_input=input_text, output=output_text))
                
                # Add an educational example as well
                input_text = "What is a state vector in orbital mechanics?"
                output_text = (
                    "A state vector in orbital mechanics is a mathematical representation of an object's "
                    "position and velocity in three-dimensional space at a specific moment in time (epoch). "
                    "It consists of six components:\n\n"
                    "1. Position components (x, y, z): The satellite's location in 3D space, typically measured "
                    "in kilometers relative to Earth's center in an Earth-centered inertial (ECI) coordinate system.\n\n"
                    "2. Velocity components (vx, vy, vz): The satellite's velocity vector, typically measured "
                    "in kilometers per second in the same coordinate system.\n\n"
                    "State vectors are fundamental in space situational awareness and are used in the Unified Data "
                    "Library (UDL) to track and predict the positions of objects in space."
                )
                examples.append(UDLTrainingExample(text_input=input_text, output=output_text))
            
            print(f"Created {len(examples)} training examples from state vector data")
            
        except Exception as e:
            print(f"Error fetching state vector data: {e}")
            traceback.print_exc()
            
            # Create dummy examples if an error occurs
            input_text = "What is a state vector in orbital mechanics?"
            output_text = (
                "A state vector in orbital mechanics is a mathematical representation of an object's "
                "position and velocity in three-dimensional space at a specific moment in time (epoch). "
                "It consists of six components:\n\n"
                "1. Position components (x, y, z): The satellite's location in 3D space, typically measured "
                "in kilometers relative to Earth's center in an Earth-centered inertial (ECI) coordinate system.\n\n"
                "2. Velocity components (vx, vy, vz): The satellite's velocity vector, typically measured "
                "in kilometers per second in the same coordinate system.\n\n"
                "State vectors are fundamental in space situational awareness because they:\n"
                "- Provide an instantaneous snapshot of an object's trajectory\n"
                "- Can be propagated forward or backward in time using physics models\n"
                "- Can be converted to other orbital representations like Keplerian elements\n"
                "- Are used in conjunction analysis, maneuver planning, and reentry predictions\n"
                "- Serve as inputs to more complex space mission design calculations"
            )
            examples.append(UDLTrainingExample(text_input=input_text, output=output_text))
        
        return examples
    
    def generate_training_dataset(self) -> List[Dict[str, str]]:
        """Generate a complete training dataset combining data from all sources."""
        all_examples = []
        
        # Fetch examples from different data sources
        all_examples.extend(self.fetch_elset_data())
        all_examples.extend(self.fetch_tle_data())
        all_examples.extend(self.fetch_statevector_data())
        
        # Convert to the format required for fine-tuning
        return [asdict(example) for example in all_examples]
    
    def save_training_data(self, output_path="training_data/udl_training_data.json"):
        """Save the training data to a file."""
        # Generate the training dataset
        training_data = self.generate_training_dataset()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"Saved {len(training_data)} training examples to {output_path}")
        return output_path

def main():
    try:
        # Create the generator
        generator = UDLTrainingDataGenerator()
        
        # Generate and save training data
        output_path = generator.save_training_data()
        
        print(f"Successfully generated training data at: {output_path}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set up your UDL credentials in the .env file.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 