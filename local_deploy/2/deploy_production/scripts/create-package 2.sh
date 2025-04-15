#!/bin/bash
# Create a complete deployment package for AstroShield

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$DEPLOY_DIR/package"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "Creating AstroShield deployment package..."

# Copy all files to the package directory
cp -r "$DEPLOY_DIR/backend" "$OUTPUT_DIR/"
cp -r "$DEPLOY_DIR/frontend" "$OUTPUT_DIR/"
cp -r "$DEPLOY_DIR/nginx" "$OUTPUT_DIR/"
cp -r "$DEPLOY_DIR/scripts" "$OUTPUT_DIR/"
cp "$DEPLOY_DIR/README.md" "$OUTPUT_DIR/"

# Create a zip archive
PACKAGE_NAME="astroshield-$(date +%Y%m%d-%H%M%S).zip"
(cd "$OUTPUT_DIR/.." && zip -r "$PACKAGE_NAME" "package")
mv "$OUTPUT_DIR/../$PACKAGE_NAME" "$DEPLOY_DIR/"

echo "Package created: $DEPLOY_DIR/$PACKAGE_NAME"
echo "Ready for deployment to astroshield.sdataplab.com"