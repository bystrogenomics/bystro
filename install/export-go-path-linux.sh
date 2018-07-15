#!/usr/bin/env bash
if [[ -n "$1" ]]
then
  DIR=$1
else
  DIR=$HOME
fi

echo -e "\n\nStoring GOPATH in .bash_profile"

(echo ""; echo 'export PATH=$PATH:/usr/local/go/bin') >> $HOME/.bash_profile;
(echo ""; echo "export GOPATH=$DIR/go") >> $HOME/.bash_profile;
echo 'export PATH=$PATH:'$DIR'/go/bin/' >> $HOME/.bash_profile

source $HOME/.bash_profile;

mkdir -p $GOPATH/src/github.com;