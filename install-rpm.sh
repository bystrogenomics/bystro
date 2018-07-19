#!/usr/bin/env bash

if [[ -n "$1" ]]
then
  INSTALL_DIR=$1
else
  INSTALL_DIR=~
fi

# LiftOver is used for the LiftOverCadd.pm package, to liftOver cadd to hg38
# and cadd's GRCh37.p13 MT to hg19
./install/install-liftover-linux.sh
./install/install-rpm-deps.sh
./install/install-lmdb-linux.sh

. ~/.bash_profile

# Perlbrew simplifies version management
./install/install-perlbrew-linux.sh $INSTALL_DIR
# Bystro is increasingly a golang progrma. Perl currently handles db fetching,
./install/install-go-linux.sh $INSTALL_DIR

. ~/.bash_profile

./install/install-perl-libs.sh
./install/install-go-packages.sh
./install/update-packages.sh

printf "\n\nDone, congrats! REMEMBER TO INCREASE ULIMIT ABOVE 1024 IF RUNNING MANY FORKS\n\n";

mkdir -p logs;
