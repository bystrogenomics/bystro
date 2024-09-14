#!/usr/bin/env bash
set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <BYSTRO_INSTALL_DIR>"
    exit 1
fi

BYSTRO_INSTALL_DIR="$1"

echo -e "\n\nInstalling Go packages (bystro-vcf, stats, snp)\n"

# Check if Go is installed
if ! command -v go >/dev/null 2>&1; then
    echo "Error: Go is not installed. Please install Go to continue."
    exit 1
fi

# Check Go version (requires at least Go 1.16)
GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//')
GO_REQUIRED_VERSION="1.20"

if [[ "$(printf '%s\n' "$GO_REQUIRED_VERSION" "$GO_VERSION" | sort -V | head -n1)" != "$GO_REQUIRED_VERSION" ]]; then
    echo "Error: Go version $GO_REQUIRED_VERSION or higher is required. You have Go $GO_VERSION."
    exit 1
fi

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

echo -e "\nInstallation complete."

# Determine the Go bin directory
if [ -n "$GOBIN" ]; then
    BINDIR="$GOBIN"
else
    if [ -n "$GOPATH" ]; then
        BINDIR="$GOPATH/bin"
    else
        BINDIR="$HOME/go/bin"
    fi
fi

# Check if the bin directory is in PATH
if [[ ":$PATH:" != *":$BINDIR:"* ]]; then
    echo "Warning: Your Go bin directory ($BINDIR) is not in your PATH."
    echo "Please add it to your PATH to use the installed commands."
fi
