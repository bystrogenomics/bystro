#!/usr/bin/env bash

if [[ -n "$1" ]]
then
  INSTALL_DIR=$1
else
  INSTALL_DIR=~
fi

# LiftOver is used for the LiftOverCadd.pm package, to liftOver cadd to hg38
# and cadd's GRCh37.p13 MT to hg19
. ./install/install-liftover-linux.sh;
. ./install/install-mac-deps.sh;
. ./install/install-lmdb-linux.sh;

. ~/.bash_profile;

# Perlbrew simplifies version management
# Mac doesn't like/build 5.28.0 without failing some tests
. ./install/install-perlbrew-linux.sh $INSTALL_DIR perl-5.26.2;
. ./install/install-perl-libs.sh;

# Bystro is increasingly a golang progrma. Perl currently handles db fetching,
. ./install/install-go-mac.sh $INSTALL_DIR;

. ~/.bash_profile;

. ./install/install-go-packages.sh;
# Not necessary for first install, but allows us to have a single entry point
# for installation and updates
. ./install/update-packages.sh;

mkdir -p logs;

printf "\n\nREMEMBER TO INCREASE ULIMIT ABOVE 1024 IF RUNNING MANY FORKS\n\n";