import os
from prometheus_client import start_http_server, Gauge
from kafka import KafkaConsumer
from datetime import datetime

KAFKA_HEALTH = Gauge('kafka_connection_health', 'Kafka cluster connection health status')
CREDENTIAL_EXPIRY = Gauge('kafka_credential_expiry_seconds', 'Seconds until credential expiration')

class KafkaHealthChecker:
    def __init__(self):
        self.consumer = KafkaConsumer(
            bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
            security_protocol='SASL_SSL',
            sasl_mechanism='SCRAM-SHA-512',
            sasl_plain_username=os.getenv('KAFKA_USERNAME'),
            sasl_plain_password=os.getenv('KAFKA_PASSWORD'),
            ssl_cafile='config/ca-cert'
        )
        start_http_server(8000)

    def check_connection(self):
        try:
            status = {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'brokers': self.consumer.bootstrap_connected()
            }
            KAFKA_HEALTH.set(1)
            return status
        except Exception as e:
            KAFKA_HEALTH.set(0)
            return {'status': 'unhealthy', 'error': str(e), 'timestamp': datetime.utcnow().isoformat()}
