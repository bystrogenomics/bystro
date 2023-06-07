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

if [[ -n "$3" ]]
then
  PROFILE=$3;
else
  PROFILE=~/.bash_profile;
fi

if [[ -n "$4" ]]
then
  NOTEST=$4;
else
  NOTEST=0;
fi

export PERLBREW_ROOT=$DIR/perl5/perlbrew;
export PERLBREW_HOME=$DIR/.perlbrew;

LOCAL_LIB="$DIR/perl5/lib/perl5"

echo -e "\n\nInstalling local perl via perlbrew into $DIR\n";

(\curl -L https://install.perlbrew.pl || \wget -O - https://install.perlbrew.pl) | bash

if ! cat $PROFILE | grep "perl5\/perlbrew\/etc\/bashrc"; then
  (echo "" ; echo "export PERLBREW_HOME=$PERLBREW_HOME") | sudo tee -a $PROFILE;
  # Not sure why this is necessary sometimes; something still wonky during perlbrew install
  (echo "" ; echo 'export PATH='$DIR'/perl5/bin/:$PATH') | sudo tee -a $PROFILE;
  (echo "" ; echo 'export PERL5LIB=$PERL5LIB:'$LOCAL_LIB) | sudo tee -a $PROFILE;
  (echo "" ; echo "source $PERLBREW_ROOT/etc/bashrc") | sudo tee -a $PROFILE;
fi

source $PERLBREW_ROOT/etc/bashrc;

cnt=$(perlbrew list | grep $VERSION | wc -l);
nCores=$(getconf _NPROCESSORS_ONLN);

if [ $cnt == 0 ]; then
  if [ $NOTEST == 0 ]; then
    perlbrew install $VERSION -j $nCores;
  else
    perlbrew install $VERSION -j $nCores -n;
  fi
fi

perlbrew switch $VERSION;
perlbrew install-cpanm;
cpanm --local-lib=$DIR/perl5 local::lib && eval $(perl -I $LOCAL_LIB -Mlocal::lib);
