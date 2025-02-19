import os
from kafka import KafkaConsumer

class SecureKafkaConsumer:
    def __init__(self, topic):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
            security_protocol='SASL_SSL',
            sasl_mechanism='SCRAM-SHA-512',
            sasl_plain_username=os.getenv('KAFKA_USERNAME'),
            sasl_plain_password=os.getenv('KAFKA_PASSWORD'),
            auto_offset_reset='earliest'
        )

    def listen(self):
        try:
            for message in self.consumer:
                print(f"Received: {message.value.decode('utf-8')}")
        except Exception as e:
            print(f"Consumer error: {str(e)}")
