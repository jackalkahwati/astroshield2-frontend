#!/bin/bash
# Script to create Kafka topics for AstroShield

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Kafka configuration
KAFKA_HOME=${KAFKA_HOME:-/opt/kafka}
BOOTSTRAP_SERVER=${BOOTSTRAP_SERVER:-localhost:9092}
ZOOKEEPER_HOST=${ZOOKEEPER_HOST:-localhost:2181}
KAFKA_TOPICS_SCRIPT=$KAFKA_HOME/bin/kafka-topics.sh

echo -e "${GREEN}===== Creating Kafka Topics for AstroShield =====${NC}"

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

# Create Subsystem 5 (Hostility Monitoring) topics
create_topic "SS5.launch.trajectory"
create_topic "SS5.launch.asat.assessment"
create_topic "SS5.reentry.prediction"

# Create Subsystem 4 (CCDM) topics
create_topic "SS4.attribution.orbital_attributes"
create_topic "SS4.attribution.shape_change"
create_topic "SS4.attribution.thermal_signature"

# Create Heartbeat topic
create_topic "AstroShield.heartbeat"

# Create Subsystem 4 (CCDM) topics
echo -e "\n${GREEN}Creating CCDM topics...${NC}"
create_topic "ccdm-events" 3 1
create_topic "ccdm-alerts" 3 1
create_topic "ccdm-reports" 3 1

# Create SDA data streaming topics
echo -e "\n${GREEN}Creating SDA data streaming topics...${NC}"
create_topic "sda-conjunction-data" 5 1
create_topic "sda-tle-updates" 3 1
create_topic "sda-ephemerides" 5 1
create_topic "sda-maneuvers" 3 1
create_topic "sda-rf-emissions" 3 1
create_topic "sda-optical-signatures" 3 1
create_topic "sda-uncorrelated-tracks" 3 1

# Create UDL integration topics
echo -e "\n${GREEN}Creating UDL integration topics...${NC}"
create_topic "udl-data-requests" 3 1
create_topic "udl-data-responses" 3 1
create_topic "udl-sync-events" 2 1

# Create Cross-Tag Correlation topics
echo -e "\n${GREEN}Creating Cross-Tag Correlation topics...${NC}"
create_topic "cross-tag-observations" 3 1
create_topic "cross-tag-matches" 3 1
create_topic "cross-tag-anomalies" 2 1

# Create RPO Shape Analysis topics
echo -e "\n${GREEN}Creating RPO Shape Analysis topics...${NC}"
create_topic "rpo-shape-data" 3 1
create_topic "rpo-shape-classifications" 2 1
create_topic "rpo-shape-anomalies" 2 1

# Create CCDM Violator List topics
echo -e "\n${GREEN}Creating CCDM Violator List topics...${NC}"
create_topic "ccdm-violators" 3 1
create_topic "ccdm-violator-updates" 3 1
create_topic "ccdm-violator-notifications" 2 1

# Create Anti-CCDM and Drag Analysis topics
echo -e "\n${GREEN}Creating Anti-CCDM and Drag Analysis topics...${NC}"
create_topic "anti-ccdm-indicators" 3 1
create_topic "drag-analysis-results" 3 1
create_topic "drag-forecasts" 3 1
create_topic "environmental-factors" 3 1
create_topic "tmdb-comparisons" 3 1

# List all topics
echo -e "\n${GREEN}Listing all topics:${NC}"
$KAFKA_TOPICS_SCRIPT --list --bootstrap-server $BOOTSTRAP_SERVER

echo -e "\n${GREEN}===== Kafka Topics Created Successfully =====${NC}" 