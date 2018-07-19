#!/usr/bin/env bash
if [[ -n "$1" ]]
then
  DIR=$1
else
  DIR=$HOME
fi

echo -e "\n\nInstalling local perl via perlbrew into $DIR\n"

(\curl -L https://install.perlbrew.pl || \wget -O - https://install.perlbrew.pl) | bash

if ! cat ~/.bash_profile | grep "perl5\/perlbrew\/etc\/bashrc"; then
  (echo "" ; echo "source $DIR/perl5/perlbrew/etc/bashrc") | sudo tee -a $HOME/.bash_profile
fi

# high sierra doesn't like 5.28.0, even with berkleydb installed (fails test 21, line length)
# hence we have a mac-specific script; could pass version as param to install-perlbrew-linux.sh
cnt=$(perlbrew list | grep perl-5.26.0 | wc -l)

if [ $cnt == 0 ]; then
  perlbrew install perl-5.26.0
fi

perlbrew switch perl-5.28.0

. $HOME/.bash_profile

curl -L https://cpanmin.us | perl - App::cpanminus

cpanm --local-lib=~/perl5 local::lib && eval $(perl -I ~/perl5/lib/perl5/ -Mlocal::lib)