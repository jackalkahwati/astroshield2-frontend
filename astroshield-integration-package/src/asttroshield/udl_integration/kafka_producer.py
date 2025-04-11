"""
Kafka Producer for UDL Integration

This module provides a Kafka producer for publishing transformed UDL data to AstroShield Kafka topics.
"""

import json
import logging
from typing import Dict, Any, Optional, List

from confluent_kafka import Producer

logger = logging.getLogger(__name__)


class AstroShieldKafkaProducer:
    """Kafka producer for publishing transformed UDL data to AstroShield Kafka topics."""

    def __init__(
        self,
        bootstrap_servers: str,
        client_id: str = "udl-integration",
        security_protocol: str = "SASL_SSL",
        sasl_mechanism: str = "PLAIN",
        sasl_username: Optional[str] = None,
        sasl_password: Optional[str] = None,
        ssl_ca_location: Optional[str] = None,
        **additional_config
    ):
        """
        Initialize the Kafka producer.

        Args:
            bootstrap_servers: Comma-separated list of Kafka broker addresses
            client_id: Client ID for the producer
            security_protocol: Security protocol (PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL)
            sasl_mechanism: SASL mechanism (PLAIN, SCRAM-SHA-256, SCRAM-SHA-512)
            sasl_username: SASL username
            sasl_password: SASL password
            ssl_ca_location: Path to CA certificate file
            **additional_config: Additional configuration parameters for the producer
        """
        config = {
            "bootstrap.servers": bootstrap_servers,
            "client.id": client_id,
        }

        # Add security configuration if provided
        if security_protocol:
            config["security.protocol"] = security_protocol

        if sasl_mechanism and sasl_username and sasl_password:
            config["sasl.mechanism"] = sasl_mechanism
            config["sasl.username"] = sasl_username
            config["sasl.password"] = sasl_password

        if ssl_ca_location:
            config["ssl.ca.location"] = ssl_ca_location

        # Add additional configuration
        config.update(additional_config)

        self.producer = Producer(config)
        logger.info(f"Initialized Kafka producer with client ID {client_id}")

    def _delivery_callback(self, err, msg):
        """
        Callback function for message delivery reports.

        Args:
            err: Error (if any)
            msg: Message that was delivered
        """
        if err:
            logger.error(f"Message delivery failed: {err}")
        else:
            topic = msg.topic()
            partition = msg.partition()
            offset = msg.offset()
            key = msg.key().decode("utf-8") if msg.key() else None
            logger.debug(
                f"Message delivered to {topic} [{partition}] at offset {offset} with key {key}"
            )

    def publish_message(
        self, topic: str, message: Dict[str, Any], key: Optional[str] = None
    ) -> None:
        """
        Publish a message to a Kafka topic.

        Args:
            topic: Kafka topic
            message: Message to publish
            key: Message key (optional)
        """
        try:
            # Convert message to JSON
            value = json.dumps(message).encode("utf-8")
            
            # Use message ID as key if not provided
            if key is None and "header" in message and "messageId" in message["header"]:
                key = message["header"]["messageId"]
            
            # Encode key if provided
            key_bytes = key.encode("utf-8") if key else None
            
            # Extract UDL references for Kafka headers
            headers = []
            
            # Check for UDL references in the header
            if "header" in message and "UDL_References" in message["header"]:
                udl_refs = message["header"]["UDL_References"]
                if udl_refs and isinstance(udl_refs, list):
                    # Create a header for each UDL reference
                    for idx, ref in enumerate(udl_refs):
                        if "topic" in ref and "id" in ref:
                            # Format: UDL_REF_{index}_TOPIC, UDL_REF_{index}_ID
                            headers.append(
                                (f"UDL_REF_{idx}_TOPIC", ref["topic"].encode("utf-8"))
                            )
                            headers.append(
                                (f"UDL_REF_{idx}_ID", ref["id"].encode("utf-8"))
                            )
                    
                    # Add the count of references
                    headers.append(
                        ("UDL_REF_COUNT", str(len(udl_refs)).encode("utf-8"))
                    )
            
            # Produce message with UDL reference headers
            self.producer.produce(
                topic=topic, 
                key=key_bytes, 
                value=value, 
                headers=headers,
                callback=self._delivery_callback
            )
            
            # Poll to handle delivery reports
            self.producer.poll(0)
            
            logger.info(f"Published message to topic {topic} with key {key} and {len(headers)} UDL reference headers")
        except Exception as e:
            logger.error(f"Error publishing message to topic {topic}: {e}")
            raise

    def publish_state_vectors(self, state_vectors: List[Dict[str, Any]]) -> None:
        """
        Publish state vectors to the appropriate AstroShield topic.
        
        Since state vectors aren't directly included in the provided topic list,
        we'll determine the most appropriate topic based on context.

        Args:
            state_vectors: List of state vectors in AstroShield format
        """
        # State vectors might be related to launch trajectories
        # If they have a launch context, use the launch.trajectory topic
        for state_vector in state_vectors:
            # Check if this is a launch-related state vector
            is_launch_related = False
            if "payload" in state_vector and "metadata" in state_vector["payload"]:
                metadata = state_vector["payload"]["metadata"]
                if "context" in metadata and "launch" in metadata["context"].lower():
                    is_launch_related = True
            
            # Choose appropriate topic based on context
            if is_launch_related:
                topic = "ss5.launch.trajectory"
            else:
                # If no specific topic exists for state vectors in the provided list,
                # we publish derived state vector data as part of heartbeat or analytics
                topic = "ss5.service.heartbeat"
                
            self.publish_message(
                topic=topic,
                message=state_vector,
                key=state_vector.get("header", {}).get("messageId")
            )

    def publish_conjunctions(self, conjunctions: List[Dict[str, Any]]) -> None:
        """
        Publish conjunctions to the AstroShield PEZ-WEZ prediction topic.

        Args:
            conjunctions: List of conjunctions in AstroShield format
        """
        for conjunction in conjunctions:
            # For conjunction data, we use the PEZ-WEZ prediction topic
            self.publish_message(
                topic="ss5.pez-wez-prediction.conjunction",
                message=conjunction,
                key=conjunction.get("header", {}).get("messageId")
            )

    def publish_launch_events(self, launch_events: List[Dict[str, Any]]) -> None:
        """
        Publish launch events to the AstroShield launch detection topic.

        Args:
            launch_events: List of launch events in AstroShield format
        """
        for launch_event in launch_events:
            # Determine if this is a detection or prediction
            is_prediction = False
            if "payload" in launch_event and "isHistorical" in launch_event["payload"]:
                is_prediction = not launch_event["payload"]["isHistorical"]
            
            # Choose appropriate topic
            if is_prediction:
                topic = "ss5.launch.prediction"
            else:
                topic = "ss5.launch.detection"
                
            self.publish_message(
                topic=topic,
                message=launch_event,
                key=launch_event.get("header", {}).get("messageId")
            )

    def publish_tracks(self, tracks: List[Dict[str, Any]]) -> None:
        """
        Publish tracks to the appropriate AstroShield topic based on context.

        Args:
            tracks: List of tracks in AstroShield format
        """
        for track in tracks:
            # Determine the track type/context to select the appropriate topic
            track_type = None
            if "payload" in track and "metadata" in track["payload"]:
                metadata = track["payload"]["metadata"]
                track_type = metadata.get("trackType", "").lower()
            
            # Choose topic based on track type
            if track_type == "eo" or track_type == "electrooptical":
                topic = "ss5.pez-wez-analysis.eo"
            elif track_type == "rf" or track_type == "radio":
                # For RF tracks, we can use the RF prediction topic
                topic = "ss5.pez-wez-prediction.rf"
            else:
                # Default to heartbeat for general tracks that don't fit elsewhere
                topic = "ss5.service.heartbeat"
                
            self.publish_message(
                topic=topic,
                message=track,
                key=track.get("header", {}).get("messageId")
            )

    def publish_ephemeris(self, ephemeris_list: List[Dict[str, Any]]) -> None:
        """
        Publish ephemeris data to the appropriate AstroShield topic.

        Args:
            ephemeris_list: List of ephemeris data in AstroShield format
        """
        for ephemeris in ephemeris_list:
            # For ephemeris data, we typically use it for prediction or trajectory
            # Check context to decide on the appropriate topic
            if "payload" in ephemeris and "metadata" in ephemeris["payload"]:
                metadata = ephemeris["payload"]["metadata"]
                if "purpose" in metadata:
                    purpose = metadata["purpose"].lower()
                    if "reentry" in purpose:
                        topic = "ss5.reentry.prediction"
                    elif "launch" in purpose:
                        topic = "ss5.launch.trajectory"
                    else:
                        # Default for general prediction
                        topic = "ss5.service.heartbeat"
                else:
                    topic = "ss5.service.heartbeat"
            else:
                topic = "ss5.service.heartbeat"
                
            self.publish_message(
                topic=topic,
                message=ephemeris,
                key=ephemeris.get("header", {}).get("messageId")
            )

    def publish_maneuvers(self, maneuvers: List[Dict[str, Any]]) -> None:
        """
        Publish maneuvers to the appropriate AstroShield topic.

        Args:
            maneuvers: List of maneuvers in AstroShield format
        """
        for maneuver in maneuvers:
            # Maneuvers might be related to separations or could indicate intent
            # Determine the appropriate topic based on context
            if "payload" in maneuver and "metadata" in maneuver["payload"]:
                metadata = maneuver["payload"]["metadata"]
                maneuver_type = metadata.get("maneuverType", "").lower()
                
                if "separation" in maneuver_type:
                    topic = "ss5.separation.detection"
                elif "intent" in maneuver_type or "suspicious" in maneuver_type:
                    topic = "ss5.launch.intent-assessment"
                else:
                    # Default topic for general maneuvers
                    topic = "ss5.service.heartbeat"
            else:
                topic = "ss5.service.heartbeat"
                
            self.publish_message(
                topic=topic,
                message=maneuver,
                key=maneuver.get("header", {}).get("messageId")
            )

    def publish_observations(self, observations: List[Dict[str, Any]]) -> None:
        """
        Publish observations to the appropriate AstroShield topic based on type.

        Args:
            observations: List of observations in AstroShield format
        """
        for observation in observations:
            # Determine observation type
            obs_type = None
            if "payload" in observation and "sensorInfo" in observation["payload"]:
                sensor_info = observation["payload"]["sensorInfo"]
                obs_type = sensor_info.get("sensorType", "").lower()
            
            # Map to the appropriate topic
            if obs_type == "eo" or obs_type == "optical" or obs_type == "electrooptical":
                topic = "ss5.pez-wez-analysis.eo"
            elif obs_type == "rf" or obs_type == "radio":
                topic = "ss5.pez-wez-prediction.rf"
            elif "weather" in obs_type:
                topic = "ss5.launch.weather-check"
            else:
                # Default for observations that don't fit elsewhere
                topic = "ss5.service.heartbeat"
                
            self.publish_message(
                topic=topic,
                message=observation,
                key=observation.get("header", {}).get("messageId")
            )

    def publish_sensor_data(self, sensor_data_list: List[Dict[str, Any]]) -> None:
        """
        Publish sensor data to the AstroShield service heartbeat topic.

        Args:
            sensor_data_list: List of sensor data in AstroShield format
        """
        for sensor_data in sensor_data_list:
            # Sensor information is typically published as heartbeats
            self.publish_message(
                topic="ss5.service.heartbeat",
                message=sensor_data,
                key=sensor_data.get("header", {}).get("messageId")
            )

    def publish_orbit_determinations(self, orbit_determinations: List[Dict[str, Any]]) -> None:
        """
        Publish orbit determinations to appropriate AstroShield topics based on context.

        Args:
            orbit_determinations: List of orbit determinations in AstroShield format
        """
        for orbit_determination in orbit_determinations:
            # Check context to determine if it's related to launch, reentry, etc.
            if "payload" in orbit_determination and "metadata" in orbit_determination["payload"]:
                metadata = orbit_determination["payload"]["metadata"]
                context = metadata.get("context", "").lower()
                
                if "launch" in context:
                    topic = "ss5.launch.trajectory"
                elif "reentry" in context:
                    topic = "ss5.reentry.prediction"
                elif "separation" in context:
                    topic = "ss5.separation.detection"
                else:
                    # Default for general orbit determinations
                    topic = "ss5.service.heartbeat"
            else:
                topic = "ss5.service.heartbeat"
                
            self.publish_message(
                topic=topic,
                message=orbit_determination,
                key=orbit_determination.get("header", {}).get("messageId")
            )

    def publish_elsets(self, elsets: List[Dict[str, Any]]) -> None:
        """
        Publish ELSETs to the appropriate AstroShield topic based on context.

        Args:
            elsets: List of ELSETs in AstroShield format
        """
        for elset in elsets:
            # ELSET data is typically used for trajectory analysis or predictions
            # Check for any context to determine the most appropriate topic
            if "payload" in elset and "metadata" in elset["payload"]:
                metadata = elset["payload"]["metadata"]
                purpose = metadata.get("purpose", "").lower()
                
                if "launch" in purpose:
                    topic = "ss5.launch.trajectory"
                elif "reentry" in purpose:
                    topic = "ss5.reentry.prediction"
                else:
                    # Default for general ELSET data
                    topic = "ss5.service.heartbeat"
            else:
                topic = "ss5.service.heartbeat"
                
            self.publish_message(
                topic=topic,
                message=elset,
                key=elset.get("header", {}).get("messageId")
            )

    def publish_weather_data(self, weather_data_list: List[Dict[str, Any]]) -> None:
        """
        Publish weather data to the AstroShield launch weather check topic.

        Args:
            weather_data_list: List of weather data in AstroShield format
        """
        for weather_data in weather_data_list:
            # Weather data is relevant for launch assessments
            self.publish_message(
                topic="ss5.launch.weather-check",
                message=weather_data,
                key=weather_data.get("header", {}).get("messageId")
            )

    def publish_sensor_tasking(self, sensor_tasking_list: List[Dict[str, Any]]) -> None:
        """
        Publish sensor tasking to the AstroShield sensor tasking topic.

        Args:
            sensor_tasking_list: List of sensor tasking in AstroShield format
        """
        for sensor_tasking in sensor_tasking_list:
            self.publish_message(
                topic="ss0.sensor.tasking",
                message=sensor_tasking,
                key=sensor_tasking.get("header", {}).get("messageId")
            )

    def publish_site_data(self, site_data_list: List[Dict[str, Any]]) -> None:
        """
        Publish site data to the AstroShield site info topic.

        Args:
            site_data_list: List of site data in AstroShield format
        """
        for site_data in site_data_list:
            self.publish_message(
                topic="ss0.site.info",
                message=site_data,
                key=site_data.get("header", {}).get("messageId")
            )

    def publish_rf_data(self, rf_data_list: List[Dict[str, Any]]) -> None:
        """
        Publish RF data to the AstroShield RF detection topic.

        Args:
            rf_data_list: List of RF data in AstroShield format
        """
        for rf_data in rf_data_list:
            self.publish_message(
                topic="ss0.rf.detection",
                message=rf_data,
                key=rf_data.get("header", {}).get("messageId")
            )

    def publish_earth_orientation_parameters(self, eop_list: List[Dict[str, Any]]) -> None:
        """
        Publish Earth orientation parameters to the AstroShield EOP topic.

        Args:
            eop_list: List of Earth orientation parameters in AstroShield format
        """
        for eop in eop_list:
            self.publish_message(
                topic="ss0.earth.orientation",
                message=eop,
                key=eop.get("header", {}).get("messageId")
            )

    def publish_solar_geomagnetic_data(self, sol_geo_list: List[Dict[str, Any]]) -> None:
        """
        Publish solar and geomagnetic data to the AstroShield solar activity topic.

        Args:
            sol_geo_list: List of solar and geomagnetic data in AstroShield format
        """
        for sol_geo in sol_geo_list:
            self.publish_message(
                topic="ss0.solar.activity",
                message=sol_geo,
                key=sol_geo.get("header", {}).get("messageId")
            )

    def publish_star_catalog(self, star_catalog_list: List[Dict[str, Any]]) -> None:
        """
        Publish star catalog to the AstroShield star catalog topic.

        Args:
            star_catalog_list: List of star catalog entries in AstroShield format
        """
        for star_catalog in star_catalog_list:
            self.publish_message(
                topic="ss0.star.catalog",
                message=star_catalog,
                key=star_catalog.get("header", {}).get("messageId")
            )

    def publish_cyber_threats(self, cyber_threats: List[Dict[str, Any]]) -> None:
        """
        Publish cyber threats to the AstroShield cyber threat topic.

        Args:
            cyber_threats: List of cyber threats in AstroShield format
        """
        for cyber_threat in cyber_threats:
            self.publish_message(
                topic="ss0.cyber.threat",
                message=cyber_threat,
                key=cyber_threat.get("header", {}).get("messageId")
            )

    def publish_link_status(self, link_status_list: List[Dict[str, Any]]) -> None:
        """
        Publish link status to the AstroShield link status topic.

        Args:
            link_status_list: List of link status records in AstroShield format
        """
        for link_status in link_status_list:
            self.publish_message(
                topic="ss0.link.status",
                message=link_status,
                key=link_status.get("header", {}).get("messageId")
            )

    def publish_comm_data(self, comm_data_list: List[Dict[str, Any]]) -> None:
        """
        Publish communications data to the AstroShield communications topic.

        Args:
            comm_data_list: List of communications data in AstroShield format
        """
        for comm_data in comm_data_list:
            self.publish_message(
                topic="ss0.comm.data",
                message=comm_data,
                key=comm_data.get("header", {}).get("messageId")
            )

    def publish_mission_ops_data(self, mission_ops_list: List[Dict[str, Any]]) -> None:
        """
        Publish mission operations data to the AstroShield mission ops topic.

        Args:
            mission_ops_list: List of mission operations data in AstroShield format
        """
        for mission_ops in mission_ops_list:
            self.publish_message(
                topic="ss0.mission.ops",
                message=mission_ops,
                key=mission_ops.get("header", {}).get("messageId")
            )

    def publish_vessel_data(self, vessel_data_list: List[Dict[str, Any]]) -> None:
        """
        Publish vessel tracking data to the AstroShield vessel topic.

        Args:
            vessel_data_list: List of vessel tracking data in AstroShield format
        """
        for vessel_data in vessel_data_list:
            self.publish_message(
                topic="ss0.vessel",
                message=vessel_data,
                key=vessel_data.get("header", {}).get("messageId")
            )

    def publish_aircraft_data(self, aircraft_data_list: List[Dict[str, Any]]) -> None:
        """
        Publish aircraft tracking data to the AstroShield aircraft topic.

        Args:
            aircraft_data_list: List of aircraft tracking data in AstroShield format
        """
        for aircraft_data in aircraft_data_list:
            self.publish_message(
                topic="ss0.aircraft",
                message=aircraft_data,
                key=aircraft_data.get("header", {}).get("messageId")
            )

    def publish_ground_imagery(self, ground_imagery_list: List[Dict[str, Any]]) -> None:
        """
        Publish ground imagery data to the AstroShield ground imagery topic.

        Args:
            ground_imagery_list: List of ground imagery data in AstroShield format
        """
        for ground_imagery in ground_imagery_list:
            self.publish_message(
                topic="ss0.ground.imagery",
                message=ground_imagery,
                key=ground_imagery.get("header", {}).get("messageId")
            )

    def publish_sky_imagery(self, sky_imagery_list: List[Dict[str, Any]]) -> None:
        """
        Publish sky imagery data to the AstroShield sky imagery topic.

        Args:
            sky_imagery_list: List of sky imagery data in AstroShield format
        """
        for sky_imagery in sky_imagery_list:
            self.publish_message(
                topic="ss0.sky.imagery",
                message=sky_imagery,
                key=sky_imagery.get("header", {}).get("messageId")
            )

    def publish_video_streaming(self, video_streaming_list: List[Dict[str, Any]]) -> None:
        """
        Publish video streaming data to the AstroShield video streaming topic.

        Args:
            video_streaming_list: List of video streaming data in AstroShield format
        """
        for video_streaming in video_streaming_list:
            self.publish_message(
                topic="ss0.video.streaming",
                message=video_streaming,
                key=video_streaming.get("header", {}).get("messageId")
            )

    def flush(self, timeout: float = 10.0) -> None:
        """
        Flush the producer to ensure all messages are delivered.

        Args:
            timeout: Maximum time to block in seconds
        """
        remaining = self.producer.flush(timeout)
        if remaining > 0:
            logger.warning(f"{remaining} messages remain in the queue after flush")
        else:
            logger.info("All messages flushed successfully") 