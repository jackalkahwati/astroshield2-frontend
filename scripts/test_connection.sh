#!/bin/bash
# Verify Kafka connectivity
${PWD}/kafka_2.13-3.9.0/bin/kafka-console-consumer.sh \
  --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" \
  --topic ss0.sensor.heartbeat \
  --from-beginning \
  --consumer.config <(envsubst < config/kafka_consumer.properties) \
  --timeout-ms 5000 \
  --property print.key=true \
  --property print.offset=true
