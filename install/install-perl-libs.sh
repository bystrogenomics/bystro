#!/usr/bin/env bash
set -e

echo -e "\n\nInstalling Perl libraries\n"

echo "PERL ROOT IN install/install-perl-libs.sh: $PERLBREW_ROOT"

# Custom installation of Data::MessagePack
install_custom_msgpack() {
    rm -rf msgpack-perl
    git clone --recursive https://github.com/bystrogenomics/msgpack-perl.git
    cd msgpack-perl
    git checkout 6fe098dd91e705b12c68d63bcb1f31c369c81e01
    cpm install -g --test .
    rm -rf msgpack-perl
}

install_custom_msgpack

ALIEN_INSTALL_TYPE=share cpm install -g --test Alien::LMDB
cpm install -g --test LMDB_File

# no --test option because it has a trivial failure related to formatting of cli help strings
cpm install -g MouseX::Getopt

cpm install -g --test --cpanfile ../perl/cpanfile