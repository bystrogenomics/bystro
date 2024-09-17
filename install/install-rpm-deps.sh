#!/usr/bin/env bash
set -e
set -o pipefail

# Ensure the script is run with root privileges
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root. Use sudo."
   exit 1
fi

# Import the MariaDB GPG key
rpm --import https://yum.mariadb.org/RPM-GPG-KEY-MariaDB

# Source OS release information
. /etc/os-release

# Determine the appropriate base URL for your OS
case "$ID" in
  "fedora")
    OS_NAME="fedora"
    OS_VERSION="${VERSION_ID}"
    ;;

  "centos"|"rhel")
    OS_NAME="centos"
    OS_VERSION="${VERSION_ID%%.*}"
    ;;

  "amzn")
    if [[ "$VERSION_ID" == "2" ]]; then
      OS_NAME="centos"
      OS_VERSION="7"
    elif [[ "$VERSION_ID" == "2023" ]]; then
      OS_NAME="rhel"
      OS_VERSION="9"
    else
      echo "Unsupported Amazon Linux version."
      exit 1
    fi
    ;;

  *)
    echo "Unsupported OS. This script supports Fedora, CentOS, RHEL, and Amazon Linux."
    exit 1
    ;;
esac

# Set the MariaDB version you wish to install
MARIADB_VERSION="10.5"

# Create the MariaDB.repo file
cat <<EOF >/etc/yum.repos.d/MariaDB.repo
# MariaDB $MARIADB_VERSION repository list - created $(date +"%F %T")
# https://mariadb.org/download/
[mariadb]
name = MariaDB
baseurl = https://yum.mariadb.org/$MARIADB_VERSION/$OS_NAME$OS_VERSION-amd64
gpgkey = https://yum.mariadb.org/RPM-GPG-KEY-MariaDB
gpgcheck = 1
EOF

# Clean the dnf cache
dnf clean all

# Install MariaDB-devel
dnf install -y MariaDB-devel

# Check if mariadb_config is installed
if command -v mariadb_config > /dev/null; then
    echo "MariaDB development libraries installed successfully."
else
    echo "Failed to install MariaDB development libraries. Please check the repository configuration."
    exit 1
fi

echo -e "\n\nInstalling RPM dependencies\n"

dnf groupinstall -y "Development Tools"

# Install all required packages
# autoconf automake make gcc perl-Data-Dumper zlib-devel bzip2 bzip2-devel xz-devel curl-devel openssl-devel libdeflate-devel are required to build htslib
# cmake required to build libdeflate-devel, which is not available on amazonlinux 2023
dnf install -y \
  autoconf automake make gcc perl-Data-Dumper zlib-devel bzip2 bzip2-devel xz-devel curl-devel openssl-devel cmake \
  openssl \
  git \
  pigz \
  unzip \
  wget \
  tar \
  libcurl-devel \
  lz4 \
  patch \
  perl \
  perl-core \
  awscli \
  pkgconf-pkg-config \
  grep

# Install Node.js 20.x
curl --silent --location https://rpm.nodesource.com/setup_20.x | bash -
dnf install -y nodejs

# Install pm2 globally using npm
npm install -g pm2

# Set the libdeflate version
LIBDEFLATE_VERSION="1.21"
LIBDEFLATE_URL="https://github.com/ebiggers/libdeflate/archive/refs/tags/v${LIBDEFLATE_VERSION}.tar.gz"

# Create a temporary directory
TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"

# Change to the temporary directory
cd $TEMP_DIR

# Download libdeflate source code
echo "Downloading libdeflate version $LIBDEFLATE_VERSION from $LIBDEFLATE_URL..."
wget $LIBDEFLATE_URL

# Extract the downloaded tar.gz file
echo "Extracting libdeflate-${LIBDEFLATE_VERSION}.tar.gz..."
tar -xvf v${LIBDEFLATE_VERSION}.tar.gz
cd libdeflate-${LIBDEFLATE_VERSION}

# Configure and build libdeflate using CMake
echo "Building libdeflate..."
cmake -B build
cmake --build build -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_INSTALL_LIBDIR=/usr/local/lib

# Change to the build directory and run make
cd build
echo "Running make..."
make

# Install libdeflate
echo "Installing libdeflate..."
make install

# Clean up by removing the temporary directory
echo "Cleaning up temporary files..."
rm -rf $TEMP_DIR

# ensure that the shared libraries are available
append_if_missing "export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH" "$PROFILE_FILE"

# ensure that any binaries are available
append_if_missing "export PATH=/usr/local/bin:\$PATH" "$PROFILE_FILE"

echo -e "\n\nAll dependencies have been installed successfully.\n"