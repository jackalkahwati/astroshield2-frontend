# Kafka Configuration Guide

## Credential Rotation
1. Request new credentials from Infrastructure_Support
2. Update `.env` file
3. Test with `bash scripts/test_connection.sh`

## Security Protocols
- Credentials never committed to git
- File permissions enforced via setup script
- Audit logging available on request

## CLI Tools Setup

1. Download Kafka 3.9.0 binaries:
```
curl -O https://archive.apache.org/dist/kafka/3.9.0/kafka_2.13-3.9.0.tgz
tar -xzf kafka_2.13-3.9.0.tgz
```

2. Verify executable permissions:
```
chmod +x kafka_2.13-3.9.0/bin/*.sh

```

## Certificate Management

1. Obtain latest CA cert from Infrastructure_Support
2. Place in `config/ca-cert`
3. Run `update-ca-trust` (Linux) or add to system keychain (MacOS)
