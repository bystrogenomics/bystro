#!/usr/bin/env bash
if [[ -n "$1" ]]
then
  DIR=$1;
else
  DIR=$HOME;
fi

if [[ -n "$2" ]]
then
  VERSION=$2;
else
  VERSION=perl-5.28.0;
fi

export PERLBREW_ROOT=$DIR/perl5/perlbrew;
export PERLBREW_HOME=/tmp/.perlbrew;

echo -e "\n\nInstalling local perl via perlbrew into $DIR\n";

(\curl -L https://install.perlbrew.pl || \wget -O - https://install.perlbrew.pl) | bash

if ! cat ~/.bash_profile | grep "perl5\/perlbrew\/etc\/bashrc"; then
  (echo "" ; echo "source $PERLBREW_ROOT/etc/bashrc") | sudo tee -a ~/.bash_profile
fi

source $PERLBREW_ROOT/etc/bashrc;

cnt=$(perlbrew list | grep $VERSION | wc -l);
nCores=$(getconf _NPROCESSORS_ONLN);

if [ $cnt == 0 ]; then
  perlbrew install $VERSION -j nCores;
fi

perlbrew switch $VERSION;

curl -L https://cpanmin.us | perl - App::cpanminus

cpanm --local-lib=~/perl5 local::lib && eval $(perl -I ~/perl5/lib/perl5/ -Mlocal::lib)
