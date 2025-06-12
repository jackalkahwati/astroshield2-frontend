#!/bin/bash
# Script to create standard Kafka topics for Space Domain Awareness System
# Based on official Welders Arc topic naming convention

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Kafka configuration
KAFKA_HOME=${KAFKA_HOME:-/opt/kafka}
BOOTSTRAP_SERVER=${BOOTSTRAP_SERVER:-localhost:9092}
ZOOKEEPER_HOST=${ZOOKEEPER_HOST:-localhost:2181}
KAFKA_TOPICS_SCRIPT=$KAFKA_HOME/bin/kafka-topics.sh

echo -e "${GREEN}===== Creating Standard SDA Kafka Topics =====${NC}"

# Check if Kafka is running
if ! nc -z localhost 9092; then
    echo -e "${RED}Error: Kafka is not running on port 9092${NC}"
    exit 1
fi

# Function to create a topic if it doesn't exist
create_topic() {
    local topic=$1
    local partitions=${2:-1}
    local replication=${3:-1}
    
    echo -e "${YELLOW}Creating topic: $topic${NC}"
    $KAFKA_TOPICS_SCRIPT --create --bootstrap-server $BOOTSTRAP_SERVER \
        --replication-factor $replication --partitions $partitions --topic $topic || {
        echo -e "${YELLOW}Topic $topic may already exist, trying to describe it${NC}"
        $KAFKA_TOPICS_SCRIPT --describe --bootstrap-server $BOOTSTRAP_SERVER --topic $topic
    }
}

# SS0 - Data Ingestion Topics
echo -e "\n${BLUE}===== Creating SS0 (Data Ingestion) Topics =====${NC}"
create_topic "ss0.data.launch-detection" 3 1
create_topic "ss0.data.manifold.request" 3 1
create_topic "ss0.data.manifold.response" 3 1
create_topic "ss0.data.rf-changes" 3 1
create_topic "ss0.data.rso-charcterization" 3 1  # Note: typo in original spec
create_topic "ss0.data.weather.contrails" 2 1
create_topic "ss0.data.weather.launch-site" 2 1
create_topic "ss0.data.weather.neutral-density" 2 1
create_topic "ss0.data.weather.realtime-orbital-density-predictions" 2 1
create_topic "ss0.data.weather.reflectivity" 2 1
create_topic "ss0.data.weather.turbulence" 2 1
create_topic "ss0.data.weather.vtec" 2 1
create_topic "ss0.data.weather.windshear" 2 1
create_topic "ss0.data.weather.windshear-jetstream-level" 2 1
create_topic "ss0.data.weather.windshear-low-level" 2 1
create_topic "ss0.launch-prediction.launch-window" 3 1
create_topic "ss0.sensor.heartbeat" 3 1
create_topic "ss0.synthetic-data.ground-imagery" 2 1
create_topic "ss0.synthetic-data.sky-imagery" 2 1
create_topic "ss0.weather.launch-site" 2 1

# SS1 - Target Modeling Topics
echo -e "\n${BLUE}===== Creating SS1 (Target Modeling) Topics =====${NC}"
create_topic "ss1.indicators.capabilities-updated" 3 1
create_topic "ss1.object-attribute-update-requested" 3 1
create_topic "ss1.request.state-vector-prediction" 3 1
create_topic "ss1.response.state-vector-prediction" 3 1
create_topic "ss1.tmdb.object-inserted" 3 1
create_topic "ss1.tmdb.object-updated" 3 1

# SS2 - State Estimation Topics
echo -e "\n${BLUE}===== Creating SS2 (State Estimation) Topics =====${NC}"
create_topic "ss2.analysis.association-message" 3 1
create_topic "ss2.analysis.score-message" 3 1
create_topic "ss2.data.elset.best-state" 3 1
create_topic "ss2.data.elset.catalog-nominee" 3 1
create_topic "ss2.data.elset.sgp4" 3 1
create_topic "ss2.data.elset.sgp4-xp" 3 1
create_topic "ss2.data.elset.uct-candidate" 3 1
create_topic "ss2.data.ephemeris" 3 1
create_topic "ss2.data.observation-track" 5 1
create_topic "ss2.data.observation-track.correlated" 5 1
create_topic "ss2.data.observation-track.true-uct" 5 1
create_topic "ss2.data.orbit-determination" 3 1
create_topic "ss2.data.state-vector" 5 1
create_topic "ss2.data.state-vector.best-state" 3 1
create_topic "ss2.data.state-vector.catalog-nominee" 3 1
create_topic "ss2.data.state-vector.uct-candidate" 3 1
create_topic "ss2.request.propagation" 3 1
create_topic "ss2.request.state-recommendation" 3 1
create_topic "ss2.requests.generic-request" 3 1
create_topic "ss2.response.propagation" 3 1
create_topic "ss2.response.state-recommendation" 3 1
create_topic "ss2.responses.generic-response" 3 1
create_topic "ss2.service.event" 2 1
create_topic "ss2.service.heartbeat" 2 1

# SS3 - Command & Control Topics
echo -e "\n${BLUE}===== Creating SS3 (Command & Control) Topics =====${NC}"
create_topic "ss3.data.accesswindow" 3 1
create_topic "ss3.data.detectionprobability" 3 1

# SS4 - CCDM Topics
echo -e "\n${BLUE}===== Creating SS4 (CCDM) Topics =====${NC}"
create_topic "ss4.attributes.orbital-attribution" 3 1
create_topic "ss4.ccdm.ccdm-db" 3 1
create_topic "ss4.ccdm.ooi" 3 1
create_topic "ss4.indicators.amr-changes" 3 1
create_topic "ss4.indicators.amr-oof" 3 1
create_topic "ss4.indicators.class-analysts-disagreement" 3 1
create_topic "ss4.indicators.imaging-maneauvers-pol-violations" 3 1  # Note: typo in original spec
create_topic "ss4.indicators.maneuvers-detected" 3 1
create_topic "ss4.indicators.maneuvers-rf-pol-oof" 3 1
create_topic "ss4.indicators.object-in-high-radiation-environment" 3 1
create_topic "ss4.indicators.object-in-unoccupied-orbit" 3 1
create_topic "ss4.indicators.object-maneuvered-coverage-gaps" 3 1
create_topic "ss4.indicators.object-not-in-un-registry" 3 1
create_topic "ss4.indicators.object-stable" 3 1
create_topic "ss4.indicators.object-threat-from-known-site" 3 1
create_topic "ss4.indicators.optical-radar-signature-mismatch" 3 1
create_topic "ss4.indicators.optical-radar-signature-oof" 3 1
create_topic "ss4.indicators.orbit-oof" 3 1
create_topic "ss4.indicators.partner-system-stimulated-object" 3 1
create_topic "ss4.indicators.proximity-events-valid-remote-sense" 3 1
create_topic "ss4.indicators.rf-detected" 3 1
create_topic "ss4.indicators.stability-changed" 3 1
create_topic "ss4.indicators.sub-sats-deployed" 3 1
create_topic "ss4.indicators.track-object-number-greater-than-launch" 3 1
create_topic "ss4.indicators.true-uct-in-eclipse" 3 1
create_topic "ss4.indicators.unknown-debris-sma-higher-than-parent" 3 1
create_topic "ss4.indicators.valid-imaging-maneuvers" 3 1
create_topic "ss4.indicators.violates-itu-fcc-filings" 3 1

# SS5 - Hostility Monitoring Topics
echo -e "\n${BLUE}===== Creating SS5 (Hostility Monitoring) Topics =====${NC}"
create_topic "ss5.launch.asat-assessment" 3 1
create_topic "ss5.launch.coplanar-assessment" 3 1
create_topic "ss5.launch.coplanar-nominal" 3 1
create_topic "ss5.launch.coplanar-opportunities" 3 1
create_topic "ss5.launch.coplanar-prediction" 3 1
create_topic "ss5.launch.detection" 5 1
create_topic "ss5.launch.fused" 5 1
create_topic "ss5.launch.intent-assessment" 3 1
create_topic "ss5.launch.nominal" 3 1
create_topic "ss5.launch.prediction" 3 1
create_topic "ss5.launch.tracklet" 3 1
create_topic "ss5.launch.trajectory" 5 1
create_topic "ss5.launch.weather-check" 2 1
create_topic "ss5.onorbit.priority-OOI" 3 1
create_topic "ss5.pez-wez-analysis.eo" 3 1
create_topic "ss5.pez-wez-prediction.conjunction" 3 1
create_topic "ss5.pez-wez-prediction.eo" 3 1
create_topic "ss5.pez-wez-prediction.grappler" 3 1
create_topic "ss5.pez-wez-prediction.kkv" 3 1
create_topic "ss5.pez-wez-prediction.rf" 3 1
create_topic "ss5.pez-wez.intent-assessment" 3 1
create_topic "ss5.polygon-closures" 3 1
create_topic "ss5.reentry.prediction" 3 1
create_topic "ss5.rpo.classification" 3 1
create_topic "ss5.rpo.image" 3 1
create_topic "ss5.rpo.intent" 3 1
create_topic "ss5.separation.detection" 3 1
create_topic "ss5.service.heartbeat" 2 1

# SS6 - Threat Assessment Topics
echo -e "\n${BLUE}===== Creating SS6 (Threat Assessment) Topics =====${NC}"
create_topic "ss6.response-recommendation.launch" 3 1
create_topic "ss6.response-recommendation.launch-asat-assessment" 3 1
create_topic "ss6.response-recommendation.launch-coplanar-risk" 3 1
create_topic "ss6.response-recommendation.launch-reentry" 3 1
create_topic "ss6.response-recommendation.on-orbit" 3 1
create_topic "ss6.risk-mitigation.optimal-maneuver" 3 1
create_topic "ss6.service.heartbeat" 2 1

# Test Environment Topics
echo -e "\n${BLUE}===== Creating Test Environment Topics =====${NC}"
create_topic "test-environment.ss0" 1 1
create_topic "test-environment.ss1" 1 1
create_topic "test-environment.ss2" 1 1
create_topic "test-environment.ss3" 1 1
create_topic "test-environment.ss4" 1 1
create_topic "test-environment.ss5" 1 1
create_topic "test-environment.ss6" 1 1

# UI Topics
echo -e "\n${BLUE}===== Creating UI Topics =====${NC}"
create_topic "ui.event" 3 1

# WA Topics (Welders Arc)
echo -e "\n${BLUE}===== Creating WA Topics =====${NC}"
create_topic "wa.new-known-object" 3 1
create_topic "wa.new-unk-object" 3 1

# List all topics
echo -e "\n${GREEN}Listing all topics:${NC}"
$KAFKA_TOPICS_SCRIPT --list --bootstrap-server $BOOTSTRAP_SERVER

echo -e "\n${GREEN}===== Standard SDA Kafka Topics Created Successfully =====${NC}"
echo -e "${YELLOW}Note: AstroShield access permissions are enforced at the application level${NC}" 