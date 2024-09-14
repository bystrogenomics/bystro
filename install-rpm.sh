#!/usr/bin/env bash
set -e
set -o pipefail

# Ensure the script is run with root privileges
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root. Use sudo."
   exit 1
fi


if [[ -n "$1" ]]; then
  HOME_DIR="$1"
else
  # Use the home directory of the invoking user, not root
  if [[ -n "$SUDO_USER" ]]; then
    HOME_DIR="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
  else
    HOME_DIR="$HOME"
  fi
fi

echo "home directory is $HOME_DIR"

INSTALL_DIR=$(pwd)

echo "install directory is $INSTALL_DIR"

PROFILE_FILE=$(./install/detect-shell-profile.sh)
GO_PLATFORM="linux-amd64"

# Install RPM dependencies
./install/install-rpm-deps.sh

# Install LiftOver
./install/install-liftover-linux.sh

# Install LMDB
./install/install-lmdb-linux.sh

# Install Perlbrew
./install/install-perlbrew-linux.sh "$HOME_DIR" perl-5.34.0

# Source Perlbrew environment for the current script execution
if [[ -f "$HOME_DIR/perl5/perlbrew/etc/bashrc" ]]; then
  source "$HOME_DIR/perl5/perlbrew/etc/bashrc"
else
  echo "Error: Perlbrew bashrc not found. Ensure Perlbrew was installed correctly."
  exit 1
fi

# Activate the new Perl version
perlbrew use perl-5.34.0

# Verify that the correct Perl version is active
CURRENT_PERL_VERSION=$(perl -v | grep "v5.34.0")
if [[ -z "$CURRENT_PERL_VERSION" ]]; then
  echo "Error: Failed to switch to Perl 5.34.0"
  exit 1
fi

# Install Perl libraries using the new Perl version
./install/install-perl-libs.sh

# Install Go
./install/install-go.sh "$HOME_DIR" "$PROFILE_FILE"

echo "Sourcing $PROFILE_FILE";

source $PROFILE_FILE;

# Install Go packages
./install/install-go-packages.sh "$INSTALL_DIR"

# Export Bystro libraries to bash_profile
./install/export-bystro-libs.sh "$INSTALL_DIR" "$PROFILE_FILE"

# Create logs directory
mkdir -p logs

echo -e "\n\nREMEMBER TO INCREASE ULIMIT ABOVE 1024 IF RUNNING MANY FORKS\n\n"

echo -e "To get started with Bystro, for instance to run Bystro Annotator: \n"
echo "Update your shell to to reflect the newly installed programs: 'source $HOME_DIR/.bash_profile'"
echo "Run Bystro Annotator: 'bystro-annotate.pl --help'"
echo -e "\n\n"
