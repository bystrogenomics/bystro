#!/usr/bin/env bash
set -e

# Ensure that INSTALL_DIR is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <BYSTRO_INSTALL_DIR> [PROFILE_FILE]"
  echo "Install directory is the directory where Bystro source files are located"
  exit 1
fi

# Get absolute path of INSTALL_DIR
BYSTRO_INSTALL_DIR="$1"
PROFILE_FILE="$2"

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

# Function to append a line to PROFILE_FILE if it doesn't already exist
append_if_missing() {
  local line="$1"
  local found=0

  # Read the profile file line by line and check if the line exists
  while IFS= read -r current_line; do
    if [ "$current_line" = "$line" ]; then
      found=1
      break
    fi
  done < "$PROFILE_FILE"

  # If the line was not found, append it to the file
  if [ $found -eq 0 ]; then
    echo -e "\n$line" >> "$PROFILE_FILE"
    echo "Added to $PROFILE_FILE: $line"
  else
    echo "Already present in $PROFILE_FILE: $line"
  fi
}


# Append entries only if they are missing
append_if_missing 'export PERL5LIB=$PERL5LIB:'"$LIB_DIR"
append_if_missing 'export PATH=$PATH:'"$PERL_BIN_DIR"