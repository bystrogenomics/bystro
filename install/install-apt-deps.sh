#!/usr/bin/env bash
set -e
set -o pipefail

# Ensure the script is run with root privileges
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root. Use sudo."
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
  pkg-config \
  grep 

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo ./aws/install

# Install Node.js 20.x
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs

# Install pm2 globally using npm
npm install -g pm2

echo -e "\n\nAll dependencies have been installed successfully.\n"
