#!/usr/bin/env bash
set -e

# This script installs Go and configures the Go environment.

# Ensure that DIR and PROFILE_FILE are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <HOME_DIR> <PROFILE_FILE>"
    exit 1
fi

HOME_DIR="$1"
PROFILE_FILE="$2"

echo -e "\n\nInstalling Go\n"

# Determine the installation directory for Go
if [ -w "/usr/local" ]; then
    INSTALL_DIR="/usr/local"
else
    INSTALL_DIR="$HOME_DIR/.local"
    mkdir -p "$INSTALL_DIR"
fi

# Check if Go is already installed
if command -v go >/dev/null 2>&1; then
    INSTALLED_GO_VERSION=$(go version | awk '{print $3}')
    echo "Go is already installed: $INSTALLED_GO_VERSION"
    echo "Skipping Go installation."
fi

# Define Go version and platform
GO_VERSION=${GO_VERSION:-"1.21.4"}
GO_PLATFORM=${GO_PLATFORM:-"linux-amd64"}
GOFILE="go${GO_VERSION}.${GO_PLATFORM}.tar.gz"

# Create temporary directory for download
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Download Go
echo "Downloading Go $GO_VERSION for $GO_PLATFORM..."
wget -q "https://dl.google.com/go/$GOFILE"

# Verify download succeeded
if [[ ! -f "$GOFILE" ]]; then
    echo "Error: Failed to download Go tarball."
    exit 1
fi

# Remove existing Go installation
if [ -d "$INSTALL_DIR/go" ]; then
    echo "Removing existing Go installation at $INSTALL_DIR/go..."
    rm -rf "$INSTALL_DIR/go"
fi

# Extract and install Go
echo "Installing Go to $INSTALL_DIR..."
tar -C "$INSTALL_DIR" -xzf "$GOFILE"

# Clean up
cd -
rm -rf "$TEMP_DIR"

# Set up environment variables
echo -e "\n\nConfiguring Go environment in $PROFILE_FILE\n"

# Ensure the profile file exists
touch "$PROFILE_FILE"

# Add Go binary directory to PATH
GO_BIN="$INSTALL_DIR/go/bin"
if ! grep -qs "$GO_BIN" "$PROFILE_FILE"; then
    echo "export PATH=\$PATH:$GO_BIN" >> "$PROFILE_FILE"
fi

# Set GOPATH
if ! grep -qs 'export GOPATH=' "$PROFILE_FILE"; then
    echo "export GOPATH=$HOME_DIR/go" >> "$PROFILE_FILE"
fi

# Add $GOPATH/bin to PATH
if ! grep -qs '\$HOME_DIR/go/bin' "$PROFILE_FILE"; then
    echo 'export PATH=$PATH:$HOME_DIR/go/bin' >> "$PROFILE_FILE"
fi

# Create GOPATH directories
mkdir -p "$HOME_DIR/go/src/github.com"

echo -e "\nGo installation complete."
echo "Please start a new shell session or run 'source $PROFILE_FILE' to apply the changes."
