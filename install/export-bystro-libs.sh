#!/usr/bin/env bash
set -e

# Ensure that INSTALL_DIR is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <PROFILE_FILE> <BYSTRO_INSTALL_DIR>"
  echo "Install directory is the directory where Bystro source files are located"
  exit 1
fi

PROFILE_FILE="$1"
BYSTRO_INSTALL_DIR="$2"

#Strip BYSTRO_INSTALL_DIR of trailing slashes
BYSTRO_INSTALL_DIR="${BYSTRO_INSTALL_DIR%/}"

echo "BYSTRO_INSTALL_DIR: $BYSTRO_INSTALL_DIR"

# Verify that BYSTRO_INSTALL_DIR has perl, perl/bin, and perl/bin/bystro-annotate.pl
if [ ! -d "$BYSTRO_INSTALL_DIR/perl" ]; then
  echo "Error: Directory $BYSTRO_INSTALL_DIR/perl does not exist."
  exit 1
fi

if [ ! -d "$BYSTRO_INSTALL_DIR/perl/bin" ]; then
  echo "Error: Directory $BYSTRO_INSTALL_DIR/perl/bin does not exist."
  exit 1
fi

if [ ! -f "$BYSTRO_INSTALL_DIR/perl/bin/bystro-annotate.pl" ]; then
  echo "Error: File $BYSTRO_INSTALL_DIR/perl/bin/bystro-annotate.pl does not exist."
  exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
# exports append_if_missing
source "$SCRIPT_DIR/utils.sh"

echo -e "\n\nExporting paths from $BYSTRO_INSTALL_DIR to $PROFILE_FILE\n"

# Verify that $INSTALL_DIR/bystro/lib exists and contains at least one .pm file
LIB_DIR="$BYSTRO_INSTALL_DIR/perl/lib"

if [ ! -d "$LIB_DIR" ]; then
  echo "Error: Directory $LIB_DIR does not exist."
  exit 1
fi

# Check if there is at least one .pm file in LIB_DIR
if ! ls "$LIB_DIR"/*.pm >/dev/null 2>&1; then
  echo "Error: No .pm files found in $LIB_DIR."
  exit 1
fi

# Verify that $INSTALL_DIR/perl/bin exists and contains at least one .pl file
PERL_BIN_DIR="$BYSTRO_INSTALL_DIR/perl/bin"

if [ ! -d "$PERL_BIN_DIR" ]; then
  echo "Error: Directory $PERL_BIN_DIR does not exist."
  exit 1
fi

# Check if there is at least one .pl file in PERL_BIN_DIR
if ! ls "$PERL_BIN_DIR"/*.pl >/dev/null 2>&1; then
  echo "Error: No .pl files found in $PERL_BIN_DIR."
  exit 1
fi

# Append entries only if they are missing
append_if_missing 'export PERL5LIB=$PERL5LIB:'"$LIB_DIR" "$PROFILE_FILE"

# Check if $PERL_BIN_DIR is in the PATH and if not, add it
if [[ ":$PATH:" != *":$PERL_BIN_DIR:"* ]]; then
  append_if_missing 'export PATH=$PATH:'"$PERL_BIN_DIR" "$PROFILE_FILE"
else
  echo "$PERL_BIN_DIR is in PATH"
fi
