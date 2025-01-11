import consul
import socket
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ServiceRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceRegistry, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if not self.initialized:
            self.consul_client = consul.Consul(host='localhost', port=8500)
            self.service_name = "astroshield"
            self.service_id = f"{self.service_name}-{socket.gethostname()}"
            self.service_port = 8000
            self.initialized = True

    def register(self):
        """Register the service with Consul"""
        try:
            self.consul_client.agent.service.register(
                name=self.service_name,
                service_id=self.service_id,
                address=socket.gethostname(),
                port=self.service_port,
                tags=['api', 'spacecraft'],
                check={
                    'http': f'http://localhost:{self.service_port}/health',
                    'interval': '10s'
                }
            )
            logger.info(f"Service registered: {self.service_id}")
        except Exception as e:
            logger.error(f"Failed to register service: {str(e)}")
            raise

    def deregister(self):
        """Deregister the service from Consul"""
        try:
            self.consul_client.agent.service.deregister(self.service_id)
            logger.info(f"Service deregistered: {self.service_id}")
        except Exception as e:
            logger.error(f"Failed to deregister service: {str(e)}")
            raise

    def get_service(self, service_name: str) -> Dict[str, Any]:
        """Get service details from Consul"""
        try:
            _, services = self.consul_client.health.service(service_name, passing=True)
            if services:
                service = services[0]['Service']
                return {
                    'id': service['ID'],
                    'name': service['Service'],
                    'address': service['Address'],
                    'port': service['Port']
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get service {service_name}: {str(e)}")
            raise
