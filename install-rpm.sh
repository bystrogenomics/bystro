#!/usr/bin/env bash
set -e
set -o pipefail

# Use the home directory of the invoking user, not root
if [[ -n "$SUDO_USER" ]]; then
  HOME_DIR="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
else
  HOME_DIR="$HOME"
fi

echo "home directory is $HOME_DIR"

BYSTRO_INSTALL_DIR=$(pwd)

LOCAL_INSTALL_DIR="$HOME_DIR/.local"
BINARY_INSTALL_DIR="$HOME_DIR/.local/bin"

echo "install directory is $BYSTRO_INSTALL_DIR"

PROFILE_FILE=$(./install/detect-shell-profile.sh "$HOME_DIR")
GO_PLATFORM="linux-amd64"

echo "PROFILE IS $PROFILE_FILE";

# Install RPM dependencies
sudo ./install/install-rpm-deps.sh

# Install LiftOver
./install/install-liftover-linux.sh "$PROFILE_FILE" "$BINARY_INSTALL_DIR" 

# Install LMDB
sudo ./install/install-lmdb-linux.sh

# Install Perlbrew
./install/install-perlbrew-linux.sh "$HOME_DIR" perl-5.34.0

# Install Go
./install/install-go.sh "$PROFILE_FILE" "LOCAL_INSTALL_DIR" "$BYSTRO_INSTALL_DIR"

# Export Bystro libraries to bash_profile
./install/export-bystro-libs.sh "$PROFILE_FILE" "$BYSTRO_INSTALL_DIR" 

# Create logs directory
mkdir -p logs

echo -e "\n\nREMEMBER TO INCREASE ULIMIT ABOVE 1024 IF RUNNING MANY FORKS\n\n"

echo -e "To get started with Bystro, for instance to run Bystro Annotator: \n"
echo "Update your shell to to reflect the newly installed programs: 'source $HOME_DIR/.bash_profile'"
echo "Run Bystro Annotator: 'bystro-annotate.pl --help'"
echo -e "\n\n"
