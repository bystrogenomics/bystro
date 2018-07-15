#!/usr/bin/env bash
if [[ -n "$1" ]]
then
  DIR=$1
else
  DIR=~
fi

echo "Installing local perl via perlbrew into $DIR"

(\curl -L https://install.perlbrew.pl || \wget -O - https://install.perlbrew.pl) | bash
(echo "" ; echo "source $DIR/perl5/perlbrew/etc/bashrc") | sudo tee -a ~/.bash_profile && source ~/.bash_profile;

# high sierra doesn't like 5.28.0, even with berkleydb installed (fails test 21, line length)
cnt=$(perlbrew list | grep perl-5.28.0 | wc -l)

if [ $cnt > 0 ]; then
  perlbrew install perl-5.26.0
fi

perlbrew switch perl-5.26.0

source ~/.bash_profile;

./install/update-cpan.sh