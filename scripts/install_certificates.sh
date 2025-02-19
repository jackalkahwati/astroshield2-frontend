#!/bin/bash
# For MacOS
security add-trusted-cert -d -k /Library/Keychains/System.keychain config/ca-cert
# For Linux
# update-ca-trust
