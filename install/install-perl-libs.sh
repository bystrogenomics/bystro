#!/usr/bin/env bash
set -e

echo -e "\n\nInstalling Perl libraries\n"

echo "PERL ROOT IN install/install-perl-libs.sh: $PERLBREW_ROOT"

# Install multiple modules at once
cpanm Capture::Tiny Mouse Path::Tiny namespace::autoclean DDP YAML::XS JSON::XS Getopt::Long::Descriptive Types::Path::Tiny Sereal MCE::Shared List::MoreUtils Log::Fast Parallel::ForkManager Cpanel::JSON::XS Mouse::Meta::Attribute::Custom::Trait::Array Net::HTTP Math::SigFigs PerlIO::utf8_strict PerlIO::gzip MouseX::SimpleConfig MouseX::ConfigFromFile Archive::Extract DBI String::Strip File::Which Hash::Merge::Simple Module::Build::XSUtil Test::LeakTrace Test::Pod Test::Exception Log::Any::Adapter File::Copy::Recursive

# Install modules that require special handling
ALIEN_INSTALL_TYPE=share cpanm Alien::LMDB
cpanm LMDB_File
cpanm MouseX::Getopt --force
cpanm IO::FDPass
cpanm Beanstalk::Client
cpanm Sys::CpuAffinity
cpanm DBD::MariaDB

# Custom installation of Data::MessagePack
install_custom_msgpack() {
    rm -rf msgpack-perl
    git clone --recursive https://github.com/bystrogenomics/msgpack-perl.git
    cd msgpack-perl
    git checkout 6fe098dd91e705b12c68d63bcb1f31c369c81e01
    perl Build.PL
    perl Build test
    perl Build install
    cd ..
    rm -rf msgpack-perl
}

install_custom_msgpack