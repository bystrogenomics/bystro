#!/usr/bin/env bash
set -e
set -o pipefail

# Ensure the script is run with root privileges
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root. Use sudo."
   exit 1
fi

# Check if mariadb_config is installed
if command -v mariadb_config > /dev/null; then
    echo "MariaDB development libraries installed successfully."
else
    echo "Failed to install MariaDB development libraries. Please check the repository configuration."
    exit 1
fi

echo -e "\n\nInstalling development tools and dependencies\n"

apt update

# Install build-essential and other required packages
apt install -y \
  build-essential \
  autoconf automake make gcc perl zlib1g-dev libbz2-dev liblzma-dev libcurl4-gnutls-dev libssl-dev \
  libmariadb-dev \
  cmake \
  git \
  pigz \
  unzip \
  wget \
  tar \
  libcurl4-openssl-dev \
  bzip2 \
  lz4 \
  patch \
  awscli \
  pkg-config \
  grep 

# Install Node.js 20.x
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs

# Install pm2 globally using npm
npm install -g pm2

echo -e "\n\nAll dependencies have been installed successfully.\n"
