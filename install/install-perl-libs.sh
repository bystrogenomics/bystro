#!/usr/bin/env bash

echo -e "\n\nInstalling perl libs\n"

echo "PERL ROOT IN install/install-perl-libs.sh: $PERLBREW_ROOT"

cpanm install Capture::Tiny
cpanm install Mouse
cpanm install Path::Tiny
cpanm install namespace::autoclean
cpanm install DDP
cpanm install YAML::XS
cpanm install JSON::XS
cpanm install Getopt::Long::Descriptive
cpanm install Types::Path::Tiny
cpanm install Sereal # extra MCE performance
cpanm install MCE::Shared
cpanm install List::MoreUtils
cpanm install Log::Fast
cpanm install Parallel::ForkManager
cpanm install Cpanel::JSON::XS
cpanm install Mouse::Meta::Attribute::Custom::Trait::Array
cpanm install Net::HTTP
cpanm install Math::SigFigs
cpanm install Search::Elasticsearch@7.713
# For now we use our own library
# Avoid issues with system liblmdb
env ALIEN_INSTALL_TYPE=share cpanm Alien::LMDB
cpanm install LMDB_File
cpanm install PerlIO::utf8_strict
cpanm install PerlIO::gzip
cpanm install MouseX::SimpleConfig
cpanm install MouseX::ConfigFromFile
# May fail installation on 5.28.0 due to minor output formatting issues
cpanm install MouseX::Getopt --force
cpanm install Archive::Extract
cpanm install DBI
cpanm install String::Strip
# Needed for fetching SQL (Utils::SqlWriter::Connection)
cpanm install DBD::mysql
cpanm install IO/FDPass.pm
cpanm install Beanstalk::Client
cpanm install Sys::CpuAffinity

cpanm install File::Which

# Needed for bin/annotate.pl
cpanm install Hash::Merge::Simple

# Custom branch of msgpack-perl that uses latest msgpack-c and
# allows prefer_float32 flag for 5-byte float storage
cpanm install Module::Build::XSUtil
cpanm install Test::LeakTrace
cpanm install Test::Pod

cpanm install Log::Any::Adapter

# A dependency of Data::MessagePack installation
cpanm install File::Copy::Recursive

cpanm --uninstall -f Data::MessagePack
rm -rf msgpack-perl
git clone --recursive https://github.com/bystrogenomics/msgpack-perl.git && cd msgpack-perl && git checkout 6fe098dd91e705b12c68d63bcb1f31c369c81e01
perl Build.PL
perl Build test
perl Build install
cd ../ && rm -rf msgpack-perl
