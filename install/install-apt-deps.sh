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
  bzip2 \
  lz4 \
  patch \
  pkg-config \
  grep 

# check whether curl is installed, because in some containers it is installed and then we get conflicts
if ! command -v curl &> /dev/null
then
    echo "curl is not installed. Installing now..."
    sudo apt install curl -y
else
    echo "curl is already installed."
fi

# Create a temporary directory
mkdir -p /tmp/awscli-install
cd /tmp/awscli-install
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install --update
cd -
rm -rf /tmp/awscli-install

echo -e "\n\nAll dependencies have been installed successfully.\n"
