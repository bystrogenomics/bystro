#!/usr/bin/env bash
set -e

echo -e "\n\nInstalling LMDB\n"

# Bystro uses LMDB as its database engine. Fast, great use of cache.

# Create a temporary directory for cloning
TEMP_DIR=$(mktemp -d)
echo "Cloning LMDB into temporary directory: $TEMP_DIR"

# Clone the LMDB repository into the temporary directory
git clone https://github.com/LMDB/lmdb.git "$TEMP_DIR/lmdb"

# Build LMDB
echo "Building LMDB..."
make -C "$TEMP_DIR/lmdb/libraries/liblmdb"

# Install LMDB
echo "Installing LMDB..."
make install -C "$TEMP_DIR/lmdb/libraries/liblmdb"

# Clean up the temporary directory
echo "Cleaning up..."
rm -rf "$TEMP_DIR"

echo "LMDB installation completed successfully."
