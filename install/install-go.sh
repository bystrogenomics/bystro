#!/usr/bin/env bash
set -e

########## This script installs Go and configures the Go environment.

# Ensure that DIR and PROFILE_FILE are provided
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <PROFILE_FILE> <GOLANG_INSTALL_DIR> <BYSTRO_GO_PROGRAMS_INSTALL_DIR> <BYSTRO_INSTALL_DIR> <GO_PLATFORM> <GO_VERSION>"
    exit 1
fi

PROFILE_FILE="$1"
INSTALL_DIR="$2"
GOPATH_PARENT_DIR="$3"
BYSTRO_INSTALL_DIR="$4"
GO_PLATFORM="$5"
GO_VERSION="$6"
GOFILE="go${GO_VERSION}.${GO_PLATFORM}.tar.gz"

# Add Go binary directory to PATH
GOPATH="$GOPATH_PARENT_DIR/go"
BYSTRO_GO_PROGRAM_INSTALL_PATH="$GOPATH/bin"
GO_INSTALL_DIR="$INSTALL_DIR/go"
GO_BIN="$GO_INSTALL_DIR/bin"

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
# exports append_if_missing
source "$SCRIPT_DIR/utils.sh"

echo -e "\n\nInstalling Go\n"

# Check if Go is already installed
if command -v go >/dev/null 2>&1; then
    INSTALLED_GO_VERSION=$(go version | awk '{print $3}')
    echo "Go is already installed: $INSTALLED_GO_VERSION"
    echo "Skipping Go installation."
else
    echo "Go is not installed. Proceeding with installation..."

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
    if [ -d "$GO_INSTALL_DIR" ]; then
        echo "Removing existing Go installation at $GO_INSTALL_DIR..."
        rm -rf "$GO_INSTALL_DIR"
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

    # Check if $GO_BIN is in the PATH and if not, add it
    if [[ ":$PATH:" != *":$GO_BIN:"* ]]; then
        export PATH="$PATH:$GO_BIN"
        append_if_missing "export PATH=\$PATH:$GO_BIN" "$PROFILE_FILE"
    fi

    echo -e "\nGo installation complete in $GO_INSTALL_DIR. Installing Bystro Go dependencies...\n"
fi

# Set GOPATH if it is not already in the profile file
export GOPATH=$GOPATH
append_if_missing "export GOPATH=$GOPATH" "$PROFILE_FILE"

if [[ ":$PATH:" != *":$BYSTRO_GO_PROGRAM_INSTALL_PATH:"* ]]; then
    export PATH="$PATH:$BYSTRO_GO_PROGRAM_INSTALL_PATH"
    append_if_missing "export PATH=\$PATH:$BYSTRO_GO_PROGRAM_INSTALL_PATH" "$PROFILE_FILE"
else
    echo "$BYSTRO_GO_PROGRAM_INSTALL_PATH in PATH, skipping"
fi
# Create GOPATH directories
mkdir -p "$GOPATH/src/github.com"

#### Install go packages

cd "$BYSTRO_INSTALL_DIR/go"
# Initialize the Go module if it doesn't exist
if [ ! -f "go.mod" ]; then
    echo "Initializing Go module..."
    go mod init bystro
fi

# Ensure dependencies are up to date
echo "Tidying up module dependencies..."
go mod tidy

# Install the local 'dosage' command
echo "Installing dosage command..."
go install ./cmd/dosage

# Install external packages
echo "Installing bystro-stats..."
go install github.com/bystrogenomics/bystro-stats@1.0.1

echo "Installing bystro-vcf..."
go install github.com/bystrogenomics/bystro-vcf@2.2.3

echo "Installing bystro-snp..."
go install github.com/bystrogenomics/bystro-snp@1.0.1

# Install yq for modifying config files
echo "Installing yq..."
go install github.com/mikefarah/yq/v2@2.4.1

# Return to the previous directory
cd -

echo -e "\nBystro Go dependency installation complete."

####

echo "Please start a new shell session or run 'source $PROFILE_FILE' to apply the changes."
