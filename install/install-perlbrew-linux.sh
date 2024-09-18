#!/usr/bin/env bash

set -e
set -o pipefail

if [[ "$#" -lt 3 ]]; then
  echo "Usage: $0 <profile_file> <installation_directory> <perl_version> [skip_tests]"
  exit 1
fi

# Parse command-line arguments or set default values
PROFILE=$1
DIR=$2
VERSION=$3
NOTEST="${4:-0}"

# If $DIR ends in / strip it
DIR="${DIR%/}"

echo "DIR IS $DIR"

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
source "$SCRIPT_DIR/utils.sh"

echo "Installation directory: $DIR"
echo "Perl version: $VERSION"
echo "Profile file: $PROFILE"
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "Skip tests during Perl installation: $NOTEST"

export PERLBREW_ROOT="$DIR/perl5/perlbrew"
export PERLBREW_HOME="$DIR/.perlbrew"
LOCAL_LIB="$DIR/perl5/lib/perl5"

echo -e "\nInstalling Perl via perlbrew into $DIR\n"

# Install perlbrew if not already installed
if ! command -v perlbrew >/dev/null 2>&1; then
  if command -v curl >/dev/null 2>&1; then
    curl -L https://install.perlbrew.pl | bash
  elif command -v wget >/dev/null 2>&1; then
    wget -O - https://install.perlbrew.pl | bash
  else
    echo "Error: Neither 'curl' nor 'wget' is installed. Please install one to proceed."
    exit 1
  fi
fi

append_if_missing "export PERLBREW_ROOT=\"$PERLBREW_ROOT\"" "$PROFILE"
append_if_missing "export PERLBREW_HOME=\"$PERLBREW_HOME\"" "$PROFILE"
append_if_missing "export PERL5LIB=\"\${PERL5LIB:+\$PERL5LIB:}$LOCAL_LIB\"" "$PROFILE"
append_if_missing "source \"$PERLBREW_ROOT/etc/bashrc\"" "$PROFILE"

# Check if $DIR/perl5/bin: is in path and if not, add it
if [[ ":$PATH:" != *":$DIR/perl5/bin:"* ]]; then
  append_if_missing "export PATH=\"$DIR/perl5/bin:\$PATH\"" "$PROFILE"
else
  echo "$DIR/perl5/bin is in PATH, skipping addition"
fi

# Source the perlbrew environment
if [[ -f "$PERLBREW_ROOT/etc/bashrc" ]]; then
  set +e
  source "$PERLBREW_ROOT/etc/bashrc"
  set -e
else
  echo "Error: perlbrew bashrc file not found at $PERLBREW_ROOT/etc/bashrc"
  exit 1
fi

# Check if the desired Perl version is already installed
if ! perlbrew list | grep -q "$VERSION"; then
  nCores=$(getconf _NPROCESSORS_ONLN)
  if [[ "$NOTEST" -eq 0 ]]; then
    echo "Installing Perl $VERSION with $nCores cores (running tests)"
    perlbrew install "$VERSION" -n -j "$nCores"
  else
    echo "Installing Perl $VERSION with $nCores cores (skipping tests)"
    perlbrew install "$VERSION" -n -j "$nCores" -n
  fi
else
  echo "Perl version $VERSION is already installed."
fi

# Switch to the installed Perl version
perlbrew switch "$VERSION"

echo "PERLBREW_ROOT is set to: $PERLBREW_ROOT"
# Install cpm
if [[ ! -x "$PERLBREW_ROOT/bin/cpm" ]]; then
  echo "Installing cpm for Perl version $VERSION"
  perlbrew install-cpm
else
  echo "cpm is already installed for Perl version $VERSION."
fi

# Install local::lib and set up environment using cpm
cpm install -g --local-lib-contained="$DIR/perl5" local::lib
eval "$(perl -I"$DIR/perl5/lib/perl5" -Mlocal::lib="$DIR/perl5")"

echo -e "\nPerlbrew installation and setup complete."

source $SCRIPT_DIR/install-perl-libs.sh

# Reminder to source the profile
echo -e "\nPlease run 'source \"$PROFILE\"' or start a new shell session to apply the changes."
