#!/usr/bin/env bash
if [[ -n "$1" ]]
then
  DIR=$1
else
  DIR=~
fi

echo "Storing GOPATH in .bash_profile"

(echo ""; echo 'export PATH=$PATH:/usr/local/go/bin') >> ~/.bash_profile;
(echo ""; echo "export GOPATH=$DIR/go") >> ~/.bash_profile;
echo 'export PATH=$PATH:'$DIR'/go/bin/' >> ~/.bash_profile && source ~/.bash_profile;

mkdir -p $GOPATH/src/github.com;