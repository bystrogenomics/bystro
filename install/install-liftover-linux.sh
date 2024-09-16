#!/usr/bin/env bash
set -e

echo -e "\n\nInstalling liftOver\n"

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <PROFILE_FILE> <BIN_INSTALL_DIR>"
    exit 1
fi

PROFILE_FILE=$1
INSTALL_DIR=$2

# LiftOver is used for the LiftOverCadd.pm package to lift over CADD scores to hg38
# and CADD's GRCh37.p13 MT to hg19.

# Function to check if liftOver is installed
check_liftover_installed() {
    if command -v liftOver >/dev/null 2>&1; then
        echo "liftOver is already installed at $(command -v liftOver)"
        return 0
    else
        return 1
    fi
}

# Install liftOver if not installed
if check_liftover_installed; then
    echo "Skipping installation of liftOver."
else
    echo "liftOver not found. Proceeding with installation..."

    mkdir -p "$INSTALL_DIR"

    wget -q http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/liftOver -O "$INSTALL_DIR/liftOver"
    chmod +x "$INSTALL_DIR/liftOver"
    # Ensure $HOME/bin is in PATH
    if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
        echo 'export PATH="$HOME/bin:$PATH"' >> "$HOME/.bashrc"
        export PATH="$HOME/bin:$PATH"
        echo "Added $HOME/bin to PATH in ~/.bashrc"
    fi

    # Make user install directory is in PROFILE_FILE


    echo "liftOver installed successfully to $INSTALL_DIR"
fi
