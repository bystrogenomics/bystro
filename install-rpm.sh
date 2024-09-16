#!/usr/bin/env bash
set -e
set -o pipefail

# Default values
DEFAULT_GO_PLATFORM="linux-amd64"
DEFAULT_GO_VERSION="1.21.4"
DEFAULT_PROFILE_FILE=$(./install/detect-shell-profile.sh "$HOME")

# Function to display usage information
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --profile-file PROFILE_FILE   Specify the shell profile file to update (default: auto-detected, e.g., ~/.bash_profile)"
    echo "  --go-platform GO_PLATFORM     Specify the Go platform (default: linux-amd64)"
    echo "  --go-version GO_VERSION       Specify the Go version (default: 1.21.4)"
    echo "  --help                        Show this help message and exit"
    echo ""
    exit 0
}

# Parse command-line arguments
PROFILE_FILE="$DEFAULT_PROFILE_FILE"
GO_PLATFORM="$DEFAULT_GO_PLATFORM"
GO_VERSION="$DEFAULT_GO_VERSION"
while [[ $# -gt 0 ]]; do
    case $1 in
        --profile-file)
            PROFILE_FILE="$2"
            shift 2
            ;;
        --go-platform)
            GO_PLATFORM="$2"
            shift 2
            ;;
        --go-version)
            GO_VERSION="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Use the home directory of the invoking user, not root
if [[ -n "$SUDO_USER" ]]; then
    HOME_DIR="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
else
    HOME_DIR="$HOME"
fi

echo "Home directory is $HOME_DIR"

BYSTRO_INSTALL_DIR=$(pwd)
LOCAL_INSTALL_DIR="$HOME_DIR/.local"
BINARY_INSTALL_DIR="$HOME_DIR/.local/bin"

echo "Install directory is $BYSTRO_INSTALL_DIR"
echo "PROFILE is $PROFILE_FILE"
echo "Go platform is $GO_PLATFORM"

# Install RPM dependencies
sudo ./install/install-rpm-deps.sh

# Install LiftOver
./install/install-liftover-linux.sh "$PROFILE_FILE" "$BINARY_INSTALL_DIR" 

# Install LMDB
sudo ./install/install-lmdb-linux.sh

# Install Perlbrew
./install/install-perlbrew-linux.sh "$PROFILE_FILE" "$HOME_DIR" perl-5.34.0

# Install Go
./install/install-go.sh "$PROFILE_FILE" "$HOME_DIR" "$LOCAL_INSTALL_DIR" "$BYSTRO_INSTALL_DIR" "$GO_PLATFORM" "$GO_VERSION"

# Export Bystro libraries to shell profile
./install/export-bystro-libs.sh "$PROFILE_FILE" "$BYSTRO_INSTALL_DIR" 

# Create logs directory
mkdir -p logs

echo -e "\n\nREMEMBER TO INCREASE ULIMIT ABOVE 1024 IF RUNNING MANY FORKS\n\n"

echo -e "To get started with Bystro, for instance to run Bystro Annotator: \n"
echo "Update your shell to reflect the newly installed programs: 'source $HOME_DIR/.bash_profile'"
echo "Run Bystro Annotator: 'bystro-annotate.pl --help'"
echo -e "\n\n"
