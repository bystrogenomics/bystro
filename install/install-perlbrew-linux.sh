#!/usr/bin/env bash
if [[ -n "$1" ]]
then
  DIR=$1
else
  DIR=$HOME
fi

echo -e "\n\nInstalling local perl via perlbrew into $DIR\n"

# Configure cpan to install locally
# Not quite sure why not local by default
./install/configure-cpan.sh

(\curl -L https://install.perlbrew.pl || \wget -O - https://install.perlbrew.pl) | bash

if ! cat ~/.bash_profile | grep "perl5\/etc\/bashrc"; then
  (echo "" ; echo "source $DIR/perl5/etc/bashrc") | sudo tee -a $HOME/.bash_profile
fi

. $HOME/.bash_profile;

cnt=$(perlbrew list | grep perl-5.28.0 | wc -l)

if [ $cnt == 0 ]; then
  perlbrew install perl-5.28.0
fi

perlbrew switch perl-5.28.0