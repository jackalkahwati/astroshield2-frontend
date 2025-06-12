# AstroShield Test Suite

Comprehensive test suite for validating AstroShield production readiness.

## Overview

The AstroShield test suite provides complete validation coverage across all system components:

- **Unit Tests**: Component-level functionality validation
- **Integration Tests**: Inter-component communication and data flow
- **Performance Tests**: Load testing and scalability validation
- **Security Tests**: Vulnerability scanning and compliance verification
- **AI/ML Tests**: Model accuracy and inference performance
- **End-to-End Tests**: Complete pipeline validation

## Quick Start

### Prerequisites

```bash
# Install test dependencies
pip install -r tests/requirements.txt

# Set up test environment
export ASTROSHIELD_TEST_ENV=test
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Running All Tests

```bash
# Run complete test suite
python tests/test_suite_runner.py

# Run with custom config
python tests/test_suite_runner.py --config tests/test_config.yaml
```

### Running Specific Test Categories

```bash
# Unit tests only
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/

# Performance tests
python -m pytest tests/performance/

# Security tests
python -m pytest tests/security/

# AI/ML tests
python -m pytest tests/ai_ml/

# End-to-end tests
python -m pytest tests/end_to_end/
```

## Test Categories

### 1. Unit Tests (`tests/unit/`)

Component-level tests for individual modules:

- **UDL Integration**: WebSocket client, message routing, latency measurement
- **Neo4j Queries**: Proximity searches, k-NN queries, spatial indexing
- **Kafka/Flink**: Stream processing, exactly-once semantics, throughput
- **AI/ML Components**: Model initialization, inference, feature extraction

**Key Tests:**
- `test_udl_integration.py`: UDL WebSocket client functionality
- `test_neo4j_proximity.py`: Graph database query performance
- `test_kafka_streaming.py`: Message broker reliability
- `test_flink_processing.py`: Stream processing accuracy

### 2. Integration Tests (`tests/integration/`)

Tests for component interactions:

- **Data Pipeline**: UDL → Kafka → Flink → Neo4j flow
- **AI/ML Pipeline**: Detection → Classification → Assessment
- **Monitoring**: Metrics collection and alerting
- **Security**: Authentication, authorization, encryption

**Key Tests:**
- `test_end_to_end_pipeline.py`: Complete data flow validation
- `test_monitoring_integration.py`: Prometheus/Grafana integration
- `test_security_integration.py`: RBAC and encryption validation

### 3. Performance Tests (`tests/performance/`)

Load and stress testing:

- **Throughput**: 50,000+ messages/second validation
- **Latency**: Sub-second end-to-end processing
- **Scalability**: 10,000+ concurrent connections
- **Resource Usage**: CPU, memory, disk, network monitoring

**Key Tests:**
- `test_load_performance.py`: Sustained and burst load scenarios
- `test_query_performance.py`: Database query optimization
- `test_ml_inference_speed.py`: Model inference benchmarks

### 4. Security Tests (`tests/security/`)

Security and compliance validation:

- **Vulnerability Scanning**: Container image analysis
- **Network Policies**: Traffic isolation verification
- **RBAC**: Permission boundary testing
- **Encryption**: TLS/mTLS validation
- **Compliance**: DoD IL-5, FIPS 140-2, STIG

**Key Tests:**
- `test_container_security.py`: Image scanning and signing
- `test_network_policies.py`: Kubernetes NetworkPolicy validation
- `test_rbac_permissions.py`: Role-based access control
- `test_encryption_compliance.py`: Data protection validation

### 5. AI/ML Tests (`tests/ai_ml/`)

Model validation and performance:

- **CCD Detector**: 94% F1 score validation
- **GNN Classifier**: 86% balanced accuracy
- **Inference Speed**: <50ms per detection
- **Robustness**: Edge case handling

**Key Tests:**
- `test_ccd_detector.py`: Spatiotemporal transformer validation
- `test_gnn_classifier.py`: Graph neural network accuracy
- `test_model_robustness.py`: Adversarial input handling

### 6. End-to-End Tests (`tests/end_to_end/`)

Complete system validation:

- **Conjunction Detection**: Real-time collision prediction
- **Threat Assessment**: CCD detection + intent classification
- **System Recovery**: Chaos engineering validation
- **Deployment Pipeline**: GitOps workflow testing

**Key Tests:**
- `test_conjunction_scenarios.py`: Collision detection accuracy
- `test_threat_detection.py`: End-to-end threat assessment
- `test_chaos_recovery.py`: MTTR validation

## Performance Requirements

The test suite validates these performance targets:

| Metric | Requirement | Test Coverage |
|--------|-------------|---------------|
| UDL Latency | <1s (P99) | ✓ WebSocket performance |
| Query Time | <200ms (P99) | ✓ Neo4j proximity queries |
| Throughput | 50k+ msg/s | ✓ Flink stream processing |
| CCD Detection | <50ms | ✓ Model inference speed |
| End-to-End | <1s | ✓ Complete pipeline |
| MTTR | <45s | ✓ Chaos engineering |

## Security Requirements

Validated security controls:

- **Container Security**: 0 critical vulnerabilities
- **Network Isolation**: Enforced NetworkPolicies
- **Access Control**: RBAC with least privilege
- **Encryption**: TLS 1.3+ for all communications
- **Compliance**: DoD IL-5 ready

## Test Configuration

Configuration via `tests/test_config.yaml`:

```yaml
test_categories:
  unit:
    enabled: true
    timeout: 300
    parallel: true

performance_thresholds:
  udl_websocket_latency_ms: 1000
  neo4j_query_time_ms: 200
  flink_throughput_msg_per_sec: 50000

security_requirements:
  max_critical_vulnerabilities: 0
  dod_il5_compliance: true
```

## CI/CD Integration

### GitHub Actions

```yaml
name: AstroShield Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Test Suite
        run: |
          python tests/test_suite_runner.py
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: astroshield_test_report_*.json
```

### GitLab CI

```yaml
test:
  stage: test
  script:
    - python tests/test_suite_runner.py
  artifacts:
    reports:
      junit: test-results.xml
    paths:
      - astroshield_test_report_*.json
```

## Test Data

### Synthetic Data Generation

The test suite includes realistic data generators:

- **Space Objects**: 1000+ objects across LEO/MEO/GEO/HEO
- **Conjunction Events**: Various miss distances and TCAs
- **CCD Scenarios**: All 7 deception tactics
- **Network Traffic**: Realistic message patterns

### Validation Datasets

Pre-generated validation sets in `tests/data/`:

- `ccd_validation_set.json`: Labeled CCD examples
- `intent_validation_set.json`: Maneuver intent labels
- `conjunction_validation_set.json`: Known conjunctions

## Monitoring Test Execution

### Real-time Monitoring

```bash
# Watch test progress
watch -n 1 'tail -n 50 test_output.log'

# Monitor resource usage
htop -p $(pgrep -f test_suite_runner)
```

### Test Metrics

The suite collects detailed metrics:

- Test execution time
- Resource utilization
- Performance benchmarks
- Error rates and types
- Coverage statistics

## Troubleshooting

### Common Issues

1. **Timeout Errors**
   ```bash
   # Increase timeout in config
   vim tests/test_config.yaml
   # Adjust: timeout: 600  # seconds
   ```

2. **Connection Failures**
   ```bash
   # Verify services are running
   docker-compose ps
   kubectl get pods -n astroshield
   ```

3. **Performance Test Failures**
   ```bash
   # Run with debug logging
   ASTROSHIELD_LOG_LEVEL=DEBUG python tests/test_suite_runner.py
   ```

### Debug Mode

```bash
# Run single test with debugging
python -m pytest tests/unit/test_neo4j_proximity.py::TestNeo4jProximityQueries::test_k_nearest_neighbor_query -v -s

# Run with coverage
python -m pytest --cov=src tests/
```

## Production Readiness Criteria

The system is considered production-ready when:

1. **Test Success Rate**: ≥95% across all categories
2. **Performance Gates**: All thresholds met
3. **Security Gates**: Zero critical vulnerabilities
4. **Required Categories**: Security, Performance, and AI/ML tests pass

## Contributing

### Adding New Tests

1. Create test file in appropriate category directory
2. Follow naming convention: `test_<component>_<aspect>.py`
3. Include docstrings and type hints
4. Add to test configuration if needed

### Test Best Practices

- Use async/await for I/O operations
- Mock external dependencies
- Include both positive and negative test cases
- Measure and assert on performance
- Clean up resources in tearDown

## Results and Reporting

### Test Reports

Generated reports include:

- **JSON Report**: Machine-readable results
- **HTML Report**: Visual dashboard
- **JUnit XML**: CI/CD integration
- **Performance Graphs**: Latency/throughput charts

### Example Report

```json
{
  "start_time": "2024-01-15T10:00:00Z",
  "end_time": "2024-01-15T10:45:00Z",
  "total_tests": 342,
  "passed_tests": 338,
  "failed_tests": 4,
  "success_rate": 0.988,
  "production_ready": true,
  "categories": {
    "unit": {"PASS": 120, "FAIL": 0},
    "integration": {"PASS": 45, "FAIL": 2},
    "performance": {"PASS": 38, "FAIL": 2},
    "security": {"PASS": 55, "FAIL": 0},
    "ai_ml": {"PASS": 42, "FAIL": 0},
    "end_to_end": {"PASS": 38, "FAIL": 0}
  }
}
```

## Support

For test suite issues or questions:

1. Check test logs in `tests/logs/`
2. Review configuration in `test_config.yaml`
3. Consult component-specific test documentation
4. Contact the AstroShield development team

---

**Last Updated**: January 2024
**Version**: 1.0.0
**Maintainers**: AstroShield Test Engineering Team 