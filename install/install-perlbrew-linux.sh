#!/usr/bin/env bash

if [[ -n "$1" ]]
then
  DIR=$1
else
  DIR=$HOME
fi

echo "Installing local perl via perlbrew into $DIR"

./install/configure-cpan.sh

(\curl -L https://install.perlbrew.pl || \wget -O - https://install.perlbrew.pl) | bash
(echo "" ; echo "source $DIR/perl5/perlbrew/etc/bashrc") | sudo tee -a $HOME/.bash_profile && source $HOME/.bash_profile;

cnt=$(perlbrew list | grep perl-5.28.0 | wc -l)

if [ $cnt > 0]; then
  perlbrew install perl-5.28.0
fi

perlbrew switch perl-5.28.0

source $HOME/.bash_profile;