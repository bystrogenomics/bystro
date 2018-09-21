#!/usr/bin/env bash

echo -e "\n\nInstalling perl libs\n"

cpanm install Mouse
cpanm install Path::Tiny
cpanm install namespace::autoclean
cpanm install DDP
cpanm install YAML::XS
cpanm install Getopt::Long::Descriptive
cpanm install Types::Path::Tiny
cpanm install MCE::Shared
cpanm install List::MoreUtils
cpanm install Log::Fast
cpanm install Parallel::ForkManager
cpanm install Cpanel::JSON::XS
cpanm install Mouse::Meta::Attribute::Custom::Trait::Array
cpanm install Net::HTTP
cpanm install Search::Elasticsearch
cpanm install Math::SigFigs
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
# Needed for bin/annotate.pl
cpanm install Hash::Merge::Simple
cpanm install Math::Round;
cpanm install Sys::CpuAffinity;
# cpanm install Math::Gauss;
cpanm install Statistics::Distributions;

# Custom branch of msgpack-perl that uses latest msgpack-c and
# allows prefer_float32 flag for 5-byte float storage
cpanm install Module::Install::XSUtil
cpanm install Module::Install::AuthorTests

# For filters
cpanm install Math::CDF

# A dependency of Data::MessagePack installation
cpanm install File::Copy::Recursive

cpanm --uninstall -f Data::MessagePack
rm -rf msgpack-perl
git clone --recursive https://github.com/akotlar/msgpack-perl.git && cd msgpack-perl
perl Makefile.PL
make test
make install
cd ../ && rm -rf msgpack-perl
