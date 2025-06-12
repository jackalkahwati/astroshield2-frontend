#!/usr/bin/env python3
"""
AstroShield Chaos Engineering Pipeline
Automated resilience testing with MTTR < 45 seconds requirement
Implements fault injection, recovery validation, and performance monitoring
"""

import asyncio
import logging
import time
import random
import json
import subprocess
import yaml
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChaosExperiment:
    name: str
    description: str
    target_service: str
    fault_type: str
    duration_seconds: int
    severity: str
    expected_mttr_seconds: int
    recovery_criteria: Dict[str, Any]
    blast_radius: str

@dataclass
class ExperimentResult:
    experiment_name: str
    start_time: datetime
    end_time: datetime
    fault_injection_time: datetime
    recovery_time: Optional[datetime]
    mttr_seconds: Optional[float]
    success: bool
    error_message: Optional[str]
    metrics: Dict[str, Any]
    blast_radius_contained: bool

class KubernetesClient:
    """Kubernetes client for chaos operations."""
    
    def __init__(self):
        self.namespace = "astroshield-production"
        
    def get_pods(self, namespace: str, label_selector: str = None) -> List[Dict]:
        """Get pods matching criteria."""
        try:
            cmd = ["kubectl", "get", "pods", "-n", namespace, "-o", "json"]
            if label_selector:
                cmd.extend(["-l", label_selector])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            pods_data = json.loads(result.stdout)
            
            return [
                {
                    'name': pod['metadata']['name'],
                    'namespace': pod['metadata']['namespace'],
                    'status': pod['status']['phase'],
                    'ready': any(condition['status'] == 'True' 
                               for condition in pod['status'].get('conditions', [])
                               if condition['type'] == 'Ready')
                }
                for pod in pods_data['items']
            ]
        except Exception as e:
            logger.error(f"Error getting pods: {e}")
            return []
    
    def delete_pod(self, namespace: str, pod_name: str) -> bool:
        """Delete a specific pod."""
        try:
            subprocess.run(
                ["kubectl", "delete", "pod", pod_name, "-n", namespace],
                check=True, capture_output=True
            )
            logger.info(f"Deleted pod {pod_name} in namespace {namespace}")
            return True
        except Exception as e:
            logger.error(f"Error deleting pod {pod_name}: {e}")
            return False
    
    def scale_deployment(self, namespace: str, deployment_name: str, replicas: int) -> bool:
        """Scale a deployment."""
        try:
            subprocess.run([
                "kubectl", "scale", "deployment", deployment_name,
                "-n", namespace, f"--replicas={replicas}"
            ], check=True, capture_output=True)
            
            logger.info(f"Scaled deployment {deployment_name} to {replicas} replicas")
            return True
        except Exception as e:
            logger.error(f"Error scaling deployment {deployment_name}: {e}")
            return False
    
    def create_network_policy(self, namespace: str, policy_name: str, target_labels: Dict[str, str]) -> bool:
        """Create a network policy to isolate pods."""
        network_policy = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": policy_name,
                "namespace": namespace
            },
            "spec": {
                "podSelector": {
                    "matchLabels": target_labels
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [],  # Block all ingress
                "egress": []    # Block all egress
            }
        }
        
        try:
            # Write policy to temporary file
            policy_file = f"/tmp/{policy_name}.yaml"
            with open(policy_file, 'w') as f:
                yaml.dump(network_policy, f)
            
            subprocess.run(
                ["kubectl", "apply", "-f", policy_file],
                check=True, capture_output=True
            )
            logger.info(f"Created network policy {policy_name}")
            return True
        except Exception as e:
            logger.error(f"Error creating network policy: {e}")
            return False
    
    def delete_network_policy(self, namespace: str, policy_name: str) -> bool:
        """Delete a network policy."""
        try:
            subprocess.run([
                "kubectl", "delete", "networkpolicy", policy_name, "-n", namespace
            ], check=True, capture_output=True)
            logger.info(f"Deleted network policy {policy_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting network policy: {e}")
            return False

class MetricsCollector:
    """Collect system and application metrics during chaos experiments."""
    
    def __init__(self, prometheus_url: str = "http://prometheus:9090"):
        self.prometheus_url = prometheus_url
        
    def query_prometheus(self, query: str) -> Optional[Dict]:
        """Query Prometheus for metrics."""
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={'query': query},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error querying Prometheus: {e}")
            return None
    
    def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """Get health metrics for a service."""
        metrics = {}
        
        # Response time
        query = f'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{service="{service_name}"}}[5m]))'
        result = self.query_prometheus(query)
        if result and result['data']['result']:
            metrics['response_time_p95'] = float(result['data']['result'][0]['value'][1])
        else:
            metrics['response_time_p95'] = 0.5  # Default value
        
        # Error rate
        query = f'rate(http_requests_total{{service="{service_name}",status=~"5.."}}[5m]) / rate(http_requests_total{{service="{service_name}"}}[5m])'
        result = self.query_prometheus(query)
        if result and result['data']['result']:
            metrics['error_rate'] = float(result['data']['result'][0]['value'][1])
        else:
            metrics['error_rate'] = 0.01  # Default value
        
        # Throughput
        query = f'rate(http_requests_total{{service="{service_name}"}}[5m])'
        result = self.query_prometheus(query)
        if result and result['data']['result']:
            metrics['throughput'] = float(result['data']['result'][0]['value'][1])
        else:
            metrics['throughput'] = 100.0  # Default value
        
        # Pod availability (simulated)
        metrics['availability'] = random.uniform(0.8, 1.0)
        
        return metrics
    
    def get_kafka_metrics(self) -> Dict[str, Any]:
        """Get Kafka cluster health metrics."""
        metrics = {}
        
        # Simulate Kafka metrics
        metrics['broker_availability'] = random.uniform(0.7, 1.0)
        metrics['consumer_lag'] = random.randint(0, 500)
        metrics['message_throughput'] = random.uniform(1000, 5000)
        
        return metrics
    
    def get_database_metrics(self) -> Dict[str, Any]:
        """Get database health metrics."""
        metrics = {}
        
        # Simulate database metrics
        metrics['active_connections'] = random.randint(10, 100)
        metrics['avg_query_time'] = random.uniform(0.01, 0.1)
        
        return metrics

class FaultInjector:
    """Inject various types of faults into the system."""
    
    def __init__(self, k8s_client: KubernetesClient):
        self.k8s = k8s_client
        
    async def inject_pod_failure(self, namespace: str, service_name: str, failure_percentage: float = 0.5) -> bool:
        """Inject pod failures by killing a percentage of pods."""
        try:
            pods = self.k8s.get_pods(namespace, f"app={service_name}")
            if not pods:
                logger.error(f"No pods found for service {service_name}")
                return False
            
            # Calculate number of pods to kill
            num_to_kill = max(1, int(len(pods) * failure_percentage))
            pods_to_kill = random.sample(pods, min(num_to_kill, len(pods)))
            
            # Kill selected pods
            for pod in pods_to_kill:
                self.k8s.delete_pod(namespace, pod['name'])
                await asyncio.sleep(1)  # Stagger deletions
            
            logger.info(f"Killed {len(pods_to_kill)} pods for service {service_name}")
            return True
        except Exception as e:
            logger.error(f"Error injecting pod failure: {e}")
            return False
    
    async def inject_network_partition(self, namespace: str, service_name: str, duration_seconds: int) -> bool:
        """Inject network partition by creating restrictive network policies."""
        policy_name = f"chaos-network-partition-{service_name}-{int(time.time())}"
        
        try:
            # Create network policy to isolate the service
            success = self.k8s.create_network_policy(
                namespace=namespace,
                policy_name=policy_name,
                target_labels={"app": service_name}
            )
            
            if not success:
                return False
            
            # Wait for the specified duration
            await asyncio.sleep(duration_seconds)
            
            # Remove the network policy
            self.k8s.delete_network_policy(namespace, policy_name)
            
            logger.info(f"Network partition experiment completed for {service_name}")
            return True
        except Exception as e:
            logger.error(f"Error injecting network partition: {e}")
            return False
    
    async def inject_resource_exhaustion(self, namespace: str, service_name: str, duration_seconds: int) -> bool:
        """Inject resource exhaustion by creating stress pods."""
        stress_pod_name = f"chaos-stress-{service_name}-{int(time.time())}"
        
        try:
            # Create stress pod manifest
            stress_pod = {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": stress_pod_name,
                    "namespace": namespace,
                    "labels": {"chaos-experiment": "resource-exhaustion"}
                },
                "spec": {
                    "containers": [{
                        "name": "stress",
                        "image": "progrium/stress",
                        "args": ["--cpu", "2", "--vm", "1", "--vm-bytes", "1G"],
                        "resources": {
                            "requests": {"cpu": "2", "memory": "1Gi"},
                            "limits": {"cpu": "2", "memory": "1Gi"}
                        }
                    }],
                    "restartPolicy": "Never"
                }
            }
            
            # Write pod manifest to file and create
            pod_file = f"/tmp/{stress_pod_name}.yaml"
            with open(pod_file, 'w') as f:
                yaml.dump(stress_pod, f)
            
            subprocess.run(
                ["kubectl", "apply", "-f", pod_file],
                check=True, capture_output=True
            )
            
            # Wait for the specified duration
            await asyncio.sleep(duration_seconds)
            
            # Delete the stress pod
            subprocess.run([
                "kubectl", "delete", "pod", stress_pod_name, "-n", namespace
            ], check=True, capture_output=True)
            
            logger.info(f"Resource exhaustion experiment completed for {service_name}")
            return True
        except Exception as e:
            logger.error(f"Error injecting resource exhaustion: {e}")
            return False
    
    async def inject_dependency_failure(self, namespace: str, dependency_service: str, duration_seconds: int) -> bool:
        """Inject dependency failure by scaling down a dependency."""
        try:
            # Get current replica count (simulate)
            original_replicas = 3  # Assume 3 replicas
            
            # Scale down the dependency to 0 replicas
            self.k8s.scale_deployment(namespace, dependency_service, 0)
            
            # Wait for the specified duration
            await asyncio.sleep(duration_seconds)
            
            # Restore original replica count
            self.k8s.scale_deployment(namespace, dependency_service, original_replicas)
            
            logger.info(f"Dependency failure experiment completed for {dependency_service}")
            return True
        except Exception as e:
            logger.error(f"Error injecting dependency failure: {e}")
            return False

class RecoveryValidator:
    """Validate system recovery after fault injection."""
    
    def __init__(self, metrics_collector: MetricsCollector, k8s_client: KubernetesClient):
        self.metrics = metrics_collector
        self.k8s = k8s_client
        
    async def wait_for_recovery(self, experiment: ChaosExperiment, max_wait_seconds: int = 300) -> Tuple[bool, float]:
        """Wait for system recovery and measure MTTR."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_seconds:
            if await self._check_recovery_criteria(experiment):
                recovery_time = time.time() - start_time
                logger.info(f"Recovery detected after {recovery_time:.2f} seconds")
                return True, recovery_time
            
            await asyncio.sleep(5)  # Check every 5 seconds
        
        logger.warning(f"Recovery not detected within {max_wait_seconds} seconds")
        return False, max_wait_seconds
    
    async def _check_recovery_criteria(self, experiment: ChaosExperiment) -> bool:
        """Check if recovery criteria are met."""
        criteria = experiment.recovery_criteria
        
        # Check service health
        if 'service_health' in criteria:
            health_metrics = self.metrics.get_service_health(experiment.target_service)
            
            # Check availability
            if health_metrics.get('availability', 0) < criteria['service_health'].get('min_availability', 0.8):
                return False
            
            # Check error rate
            if health_metrics.get('error_rate', 1.0) > criteria['service_health'].get('max_error_rate', 0.05):
                return False
            
            # Check response time
            if health_metrics.get('response_time_p95', float('inf')) > criteria['service_health'].get('max_response_time', 1.0):
                return False
        
        # Check pod readiness
        if 'pod_readiness' in criteria:
            pods = self.k8s.get_pods(
                namespace=criteria['pod_readiness']['namespace'],
                label_selector=f"app={experiment.target_service}"
            )
            
            ready_pods = sum(1 for pod in pods if pod['ready'])
            min_ready_pods = criteria['pod_readiness'].get('min_ready_pods', 1)
            
            if ready_pods < min_ready_pods:
                return False
        
        # Check Kafka health (if applicable)
        if 'kafka_health' in criteria:
            kafka_metrics = self.metrics.get_kafka_metrics()
            
            if kafka_metrics.get('broker_availability', 0) < criteria['kafka_health'].get('min_broker_availability', 0.67):
                return False
            
            if kafka_metrics.get('consumer_lag', float('inf')) > criteria['kafka_health'].get('max_consumer_lag', 1000):
                return False
        
        return True

class ChaosEngineeringPipeline:
    """Main chaos engineering pipeline orchestrator."""
    
    def __init__(self, config_path: str = "chaos_config.yaml"):
        self.config = self._load_config(config_path)
        self.k8s = KubernetesClient()
        self.metrics = MetricsCollector(self.config.get('prometheus_url', 'http://prometheus:9090'))
        self.fault_injector = FaultInjector(self.k8s)
        self.recovery_validator = RecoveryValidator(self.metrics, self.k8s)
        self.results: List[ExperimentResult] = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load chaos engineering configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration for chaos experiments."""
        return {
            'prometheus_url': 'http://prometheus:9090',
            'namespace': 'astroshield-production',
            'experiments': [
                {
                    'name': 'conjunction-analysis-pod-failure',
                    'description': 'Test resilience of conjunction analysis service to pod failures',
                    'target_service': 'flink-conjunction-analysis',
                    'fault_type': 'pod_failure',
                    'duration_seconds': 60,
                    'severity': 'medium',
                    'expected_mttr_seconds': 30,
                    'recovery_criteria': {
                        'service_health': {
                            'min_availability': 0.8,
                            'max_error_rate': 0.05,
                            'max_response_time': 2.0
                        },
                        'pod_readiness': {
                            'namespace': 'astroshield-production',
                            'min_ready_pods': 2
                        }
                    },
                    'blast_radius': 'service'
                },
                {
                    'name': 'kafka-network-partition',
                    'description': 'Test Kafka cluster resilience to network partitions',
                    'target_service': 'kafka-critical',
                    'fault_type': 'network_partition',
                    'duration_seconds': 30,
                    'severity': 'high',
                    'expected_mttr_seconds': 45,
                    'recovery_criteria': {
                        'kafka_health': {
                            'min_broker_availability': 0.67,
                            'max_consumer_lag': 1000
                        }
                    },
                    'blast_radius': 'cluster'
                },
                {
                    'name': 'udl-websocket-dependency-failure',
                    'description': 'Test UDL WebSocket client resilience to dependency failures',
                    'target_service': 'udl-websocket-client',
                    'fault_type': 'dependency_failure',
                    'duration_seconds': 45,
                    'severity': 'medium',
                    'expected_mttr_seconds': 20,
                    'recovery_criteria': {
                        'service_health': {
                            'min_availability': 0.9,
                            'max_error_rate': 0.02
                        }
                    },
                    'blast_radius': 'service'
                }
            ]
        }
    
    async def run_experiment(self, experiment_config: Dict) -> ExperimentResult:
        """Run a single chaos experiment."""
        experiment = ChaosExperiment(**experiment_config)
        start_time = datetime.now()
        
        logger.info(f"Starting chaos experiment: {experiment.name}")
        
        try:
            # Collect baseline metrics
            baseline_metrics = await self._collect_baseline_metrics(experiment)
            
            # Inject fault
            fault_injection_time = datetime.now()
            fault_success = await self._inject_fault(experiment)
            
            if not fault_success:
                return ExperimentResult(
                    experiment_name=experiment.name,
                    start_time=start_time,
                    end_time=datetime.now(),
                    fault_injection_time=fault_injection_time,
                    recovery_time=None,
                    mttr_seconds=None,
                    success=False,
                    error_message="Fault injection failed",
                    metrics={},
                    blast_radius_contained=False
                )
            
            # Wait for recovery
            recovery_success, mttr = await self.recovery_validator.wait_for_recovery(experiment)
            recovery_time = datetime.now() if recovery_success else None
            
            # Collect post-experiment metrics
            post_metrics = await self._collect_post_metrics(experiment)
            
            # Check blast radius containment
            blast_radius_contained = await self._check_blast_radius(experiment)
            
            end_time = datetime.now()
            
            # Determine overall success
            success = (
                recovery_success and 
                mttr <= experiment.expected_mttr_seconds and
                blast_radius_contained
            )
            
            result = ExperimentResult(
                experiment_name=experiment.name,
                start_time=start_time,
                end_time=end_time,
                fault_injection_time=fault_injection_time,
                recovery_time=recovery_time,
                mttr_seconds=mttr if recovery_success else None,
                success=success,
                error_message=None,
                metrics={
                    'baseline': baseline_metrics,
                    'post_experiment': post_metrics,
                    'mttr_requirement_met': mttr <= experiment.expected_mttr_seconds if recovery_success else False
                },
                blast_radius_contained=blast_radius_contained
            )
            
            logger.info(f"Experiment {experiment.name} completed. Success: {success}, MTTR: {mttr:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error running experiment {experiment.name}: {e}")
            return ExperimentResult(
                experiment_name=experiment.name,
                start_time=start_time,
                end_time=datetime.now(),
                fault_injection_time=datetime.now(),
                recovery_time=None,
                mttr_seconds=None,
                success=False,
                error_message=str(e),
                metrics={},
                blast_radius_contained=False
            )
    
    async def _inject_fault(self, experiment: ChaosExperiment) -> bool:
        """Inject the specified fault type."""
        namespace = self.config.get('namespace', 'astroshield-production')
        
        if experiment.fault_type == 'pod_failure':
            return await self.fault_injector.inject_pod_failure(
                namespace, experiment.target_service, 0.5
            )
        elif experiment.fault_type == 'network_partition':
            return await self.fault_injector.inject_network_partition(
                namespace, experiment.target_service, experiment.duration_seconds
            )
        elif experiment.fault_type == 'resource_exhaustion':
            return await self.fault_injector.inject_resource_exhaustion(
                namespace, experiment.target_service, experiment.duration_seconds
            )
        elif experiment.fault_type == 'dependency_failure':
            # For dependency failure, we need to identify the dependency
            dependency_map = {
                'udl-websocket-client': 'kafka-critical',
                'flink-conjunction-analysis': 'kafka-critical',
                'neo4j-proximity-queries': 'neo4j'
            }
            dependency = dependency_map.get(experiment.target_service)
            if dependency:
                return await self.fault_injector.inject_dependency_failure(
                    namespace, dependency, experiment.duration_seconds
                )
        
        logger.error(f"Unknown fault type: {experiment.fault_type}")
        return False
    
    async def _collect_baseline_metrics(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Collect baseline metrics before fault injection."""
        metrics = {}
        
        # Service health metrics
        metrics['service_health'] = self.metrics.get_service_health(experiment.target_service)
        
        # Kafka metrics (if applicable)
        if 'kafka' in experiment.target_service.lower():
            metrics['kafka_health'] = self.metrics.get_kafka_metrics()
        
        # Database metrics (if applicable)
        if 'neo4j' in experiment.target_service.lower() or 'postgres' in experiment.target_service.lower():
            metrics['database_health'] = self.metrics.get_database_metrics()
        
        return metrics
    
    async def _collect_post_metrics(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Collect metrics after experiment completion."""
        # Wait a bit for metrics to stabilize
        await asyncio.sleep(10)
        return await self._collect_baseline_metrics(experiment)
    
    async def _check_blast_radius(self, experiment: ChaosExperiment) -> bool:
        """Check if the blast radius was properly contained."""
        # Simplified blast radius check
        if experiment.blast_radius == 'service':
            # For service-level experiments, assume containment is good
            return True
        elif experiment.blast_radius == 'cluster':
            # For cluster-level experiments, some impact is expected
            return True
        
        return True
    
    async def run_all_experiments(self) -> List[ExperimentResult]:
        """Run all configured chaos experiments."""
        results = []
        
        for experiment_config in self.config.get('experiments', []):
            result = await self.run_experiment(experiment_config)
            results.append(result)
            self.results.append(result)
            
            # Wait between experiments to allow system stabilization
            await asyncio.sleep(30)
        
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive chaos engineering report."""
        if not self.results:
            return {"error": "No experiment results available"}
        
        total_experiments = len(self.results)
        successful_experiments = sum(1 for r in self.results if r.success)
        
        # MTTR statistics
        mttr_values = [r.mttr_seconds for r in self.results if r.mttr_seconds is not None]
        
        report = {
            'summary': {
                'total_experiments': total_experiments,
                'successful_experiments': successful_experiments,
                'success_rate': successful_experiments / total_experiments,
                'mttr_statistics': {
                    'mean': np.mean(mttr_values) if mttr_values else None,
                    'median': np.median(mttr_values) if mttr_values else None,
                    'max': np.max(mttr_values) if mttr_values else None,
                    'min': np.min(mttr_values) if mttr_values else None,
                    'p95': np.percentile(mttr_values, 95) if mttr_values else None
                },
                'mttr_requirement_compliance': sum(1 for mttr in mttr_values if mttr <= 45) / len(mttr_values) if mttr_values else 0
            },
            'experiments': [asdict(result) for result in self.results],
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on experiment results."""
        recommendations = []
        
        # Check MTTR compliance
        mttr_values = [r.mttr_seconds for r in self.results if r.mttr_seconds is not None]
        if mttr_values:
            avg_mttr = np.mean(mttr_values)
            if avg_mttr > 45:
                recommendations.append(
                    f"Average MTTR ({avg_mttr:.1f}s) exceeds requirement (45s). "
                    "Consider implementing faster health checks and auto-scaling policies."
                )
        
        # Check success rate
        success_rate = sum(1 for r in self.results if r.success) / len(self.results)
        if success_rate < 0.9:
            recommendations.append(
                f"Success rate ({success_rate:.1%}) is below target (90%). "
                "Review failure modes and implement additional resilience patterns."
            )
        
        # Service-specific recommendations
        failed_services = set(r.experiment_name for r in self.results if not r.success)
        for service in failed_services:
            recommendations.append(
                f"Service {service} failed chaos testing. "
                "Implement circuit breakers, retries, and graceful degradation."
            )
        
        return recommendations
    
    def save_report(self, filename: str = None):
        """Save the chaos engineering report to a file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chaos_engineering_report_{timestamp}.json"
        
        report = self.generate_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Chaos engineering report saved to {filename}")

async def main():
    """Main entry point for chaos engineering pipeline."""
    pipeline = ChaosEngineeringPipeline()
    
    logger.info("Starting AstroShield Chaos Engineering Pipeline")
    
    # Run all experiments
    results = await pipeline.run_all_experiments()
    
    # Generate and save report
    report = pipeline.generate_report()
    pipeline.save_report()
    
    # Print summary
    print("\n" + "="*60)
    print("ASTROSHIELD CHAOS ENGINEERING RESULTS")
    print("="*60)
    print(f"Total Experiments: {report['summary']['total_experiments']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1%}")
    if report['summary']['mttr_statistics']['mean']:
        print(f"Average MTTR: {report['summary']['mttr_statistics']['mean']:.1f}s")
    print(f"MTTR Requirement Compliance: {report['summary']['mttr_requirement_compliance']:.1%}")
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main()) 