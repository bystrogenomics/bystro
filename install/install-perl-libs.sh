#!/usr/bin/env bash
set -e

echo -e "\n\nInstalling Perl libraries\n"

echo "PERL ROOT IN install/install-perl-libs.sh: $PERLBREW_ROOT"

BASEDIR=$(dirname $0)
echo "install-perl-libs.sh basedir: $BASEDIR"

# Install modules that require special handling
ALIEN_INSTALL_TYPE=share cpm install -g --test Alien::LMDB
cpm install -g --test LMDB_File
cpm install -g MouseX::Getopt # fails due to differences from expected output; unimportant
cpm install -g --test IO::FDPass
cpm install -g --test Beanstalk::Client
cpm install -g --test Sys::CpuAffinity
cpm install -g --test DBD::MariaDB

# Custom installation of Data::MessagePack
install_custom_msgpack() {
    rm -rf msgpack-perl
    git clone --recursive https://github.com/bystrogenomics/msgpack-perl.git
    cd msgpack-perl
    git checkout 6fe098dd91e705b12c68d63bcb1f31c369c81e01
    cpm install -g --test .
    rm -rf msgpack-perl
    cd -
}

install_custom_msgpack

# Install the rest of the modules
(
  cd "$BASEDIR" || exit 1
  cd ../ || exit 1
  echo "Attempting to install remaining requirements from bystro/perl/cpanfile"
  cpm install -g --test --cpanfile perl/cpanfile
)