#!/usr/bin/env bash
set -e
set -o pipefail

# Ensure the script is run with root privileges
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root. Use sudo."
   exit 1
fi


if [[ -n "$1" ]]; then
  INSTALL_DIR="$1"
else
  # Use the home directory of the invoking user, not root
  if [[ -n "$SUDO_USER" ]]; then
    INSTALL_DIR="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
  else
    INSTALL_DIR="$HOME"
  fi
fi

# Install RPM dependencies
./install/install-rpm-deps.sh

# Install LiftOver
./install/install-liftover-linux.sh

# Install LMDB
./install/install-lmdb-linux.sh

# Install Perlbrew
./install/install-perlbrew-linux.sh "$INSTALL_DIR" perl-5.34.0

# Source Perlbrew environment for the current script execution
if [[ -f "$INSTALL_DIR/perl5/perlbrew/etc/bashrc" ]]; then
  source "$INSTALL_DIR/perl5/perlbrew/etc/bashrc"
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
./install/install-go-linux.sh "$INSTALL_DIR"

# Source Go environment variables if necessary
if [[ -f "$INSTALL_DIR/.bash_profile" ]]; then
  source "$INSTALL_DIR/.bash_profile"
fi

# Install Go packages
./install/install-go-packages.sh

# Update packages
./install/update-packages.sh

# Export Bystro libraries to bash_profile
./install/export-bystro-libs.sh "$INSTALL_DIR/.bash_profile"

# Create logs directory
mkdir -p logs

echo -e "\n\nREMEMBER TO INCREASE ULIMIT ABOVE 1024 IF RUNNING MANY FORKS\n\n"

echo -e "If running for the first time, you may need to source the following in your shell:\n"
echo "source $INSTALL_DIR/perl5/perlbrew/etc/bashrc"
echo "source $INSTALL_DIR/.bash_profile"
echo -e "\nAlternatively, add the above lines to your ~/.bash_profile or ~/.bashrc to make the changes permanent.\n"
