#!/usr/bin/env bash

set -e
set -o pipefail

# Parse command-line arguments or set default values
DIR="${1:-$HOME}"
VERSION="${2:-perl-5.28.0}"
PROFILE="${3:-$HOME/.bash_profile}"
NOTEST="${4:-0}"

echo "Installation directory: $DIR"
echo "Perl version: $VERSION"
echo "Profile file: $PROFILE"
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

# Update profile if perlbrew is not already sourced
if ! grep -q "source \"$PERLBREW_ROOT/etc/bashrc\"" "$PROFILE"; then
  {
    echo ""
    echo "export PERLBREW_ROOT=\"$PERLBREW_ROOT\""
    echo "export PERLBREW_HOME=\"$PERLBREW_HOME\""
    echo "export PATH=\"$DIR/perl5/bin:\$PATH\""
    echo "export PERL5LIB=\"\${PERL5LIB:+\$PERL5LIB:}$LOCAL_LIB\""
    echo "source \"$PERLBREW_ROOT/etc/bashrc\""
  } >> "$PROFILE"
  echo "Perlbrew environment variables added to $PROFILE"
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
    perlbrew install "$VERSION" -j "$nCores"
  else
    echo "Installing Perl $VERSION with $nCores cores (skipping tests)"
    perlbrew install "$VERSION" -j "$nCores" -n
  fi
else
  echo "Perl version $VERSION is already installed."
fi

# Switch to the installed Perl version
perlbrew switch "$VERSION"

echo "PERLBREW_ROOT is set to: $PERLBREW_ROOT"
# Install cpanm
if [[ ! -x "$PERLBREW_ROOT/perls/$VERSION/bin/cpanm" ]]; then
  echo "Installing cpanm for Perl version $VERSION"
  perlbrew install-cpanm
else
  echo "cpanm is already installed for Perl version $VERSION."
fi

# Install local::lib and set up environment
cpanm --local-lib="$DIR/perl5" local::lib
eval "$(perl -I"$LOCAL_LIB" -Mlocal::lib)"

echo -e "\nPerlbrew installation and setup complete."

# Reminder to source the profile
echo -e "\nPlease run 'source \"$PROFILE\"' or start a new shell session to apply the changes."
