#!/usr/bin/env bash

if [[ -n "$1" ]]
then
  PROFILE=$1;
else
  PROFILE=~/.bash_profile;
fi

DIR=$(pwd)

echo -e "\n\nExporting $DIR/lib and $DIR/bin to $PROFILE\n";

if ! cat $PROFILE | grep "$DIR/bin"; then
  (echo "" ; echo 'export PERL5LIB=$PERL5LIB:'$DIR'/lib') | tee -a $PROFILE
  # Not 100% sure why this is necessary; something still wonky during perlbrew install
  (echo "" ; echo 'export PATH=$PATH:'$DIR'/bin') | tee -a $PROFILE
fi