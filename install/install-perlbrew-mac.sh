#!/usr/bin/env bash
if [[ -n "$1" ]]
then
  DIR=$1
else
  DIR=~
fi

echo -e "\n\nInstalling local perl via perlbrew into $DIR\n"

(\curl -L https://install.perlbrew.pl || \wget -O - https://install.perlbrew.pl) | bash
(echo "" ; echo "source $DIR/perl5/perlbrew/etc/bashrc") | sudo tee -a $HOME/.bash_profile && source $HOME/.bash_profile;

# high sierra doesn't like 5.28.0, even with berkleydb installed (fails test 21, line length)
cnt=$(perlbrew list | grep perl-5.26.0 | wc -l)

if [ $cnt > "0" ]; then
  perlbrew install perl-5.26.0
fi

perlbrew switch perl-5.26.0

# Configure cpan to install locally
# Not quite sure why not local by default
./install/configure-cpan.sh

source $HOME/.bash_profile;
