#!/usr/bin/env bash
set -e
set -o pipefail

# Ensure the script is run with root privileges
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root. Use sudo."
   exit 1
fi

echo -e "\n\nUpdating system packages\n"
dnf update -y

echo -e "\n\nInstalling RPM dependencies\n"

# Install all required packages
dnf install -y \
  gcc \
  openssl \
  openssl-devel \
  git \
  pigz \
  unzip \
  wget \
  tar \
  libcurl-devel \
  bzip2 \
  lz4 \
  patch \
  perl \
  perl-core \
  awscli \
  pkgconf-pkg-config

# Install Node.js 20.x
curl --silent --location https://rpm.nodesource.com/setup_20.x | bash -
dnf install -y nodejs

# Install pm2 globally using npm
npm install -g pm2

# Install cpanminus directly using cURL
curl -L https://cpanmin.us | perl - App::cpanminus

echo -e "\n\nAll dependencies have been installed successfully.\n"