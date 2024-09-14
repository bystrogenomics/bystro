#!/usr/bin/env bash
set -e

echo -e "\n\nInstalling liftOver\n"

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

    # Determine installation directory
    if [ -w "/usr/local/bin" ]; then
        INSTALL_DIR="/usr/local/bin"
    else
        INSTALL_DIR="$HOME/bin"
        mkdir -p "$INSTALL_DIR"
        # Ensure $HOME/bin is in PATH
        if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
            echo 'export PATH="$HOME/bin:$PATH"' >> "$HOME/.bashrc"
            export PATH="$HOME/bin:$PATH"
            echo "Added $HOME/bin to PATH in ~/.bashrc"
        fi
    fi

    # Download and install liftOver
    wget -q http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/liftOver -O "$INSTALL_DIR/liftOver"
    chmod +x "$INSTALL_DIR/liftOver"

    echo "liftOver installed successfully to $INSTALL_DIR"
fi
