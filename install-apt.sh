#!/usr/bin/env bash

if [[ -n "$1" ]]
then
  INSTALL_DIR=$1
else
  INSTALL_DIR=~
fi

./install/install-apt-deps.sh

./install/install-lmdb-linux.sh

# Bystro is increasingly a golang progrma. Perl currently handles db fetching,
./install/install-go-linux.sh $INSTALL_DIR
./install/install-go-packages.sh

# LiftOver is used for the LiftOverCadd.pm package, to liftOver cadd to hg38
# and cadd's GRCh37.p13 MT to hg19
./install/install-liftover-linux.sh

# Perlbrew simplifies version management
./install/install-perlbrew-linux.sh $INSTALL_DIR

./install/install-perl-libs.sh

# Not necessary for first install, but allows us to have a single entry point
# for installation and updates
./install/update-packages.sh

printf "\n\nDone, congrats! REMEMBER TO INCREASE ULIMIT ABOVE 1024 IF RUNNING MANY FORKS\n\n";

mkdir -p logs;
