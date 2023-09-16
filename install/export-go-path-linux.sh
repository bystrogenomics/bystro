#!/usr/bin/env bash
if [[ -n $1 ]]; then
	DIR=$1
else
	DIR=~
fi

if [[ -n $2 ]]; then
	PROFILE=$2
else
	PROFILE=~/.bash_profile
fi

echo -e "\n\nStoring GOPATH in $PROFILE\n"

(
	echo ""
	echo 'export PATH=$PATH:/usr/local/go/bin'
) >>$PROFILE
(
	echo ""
	echo "export GOPATH=$DIR/go"
) >>$PROFILE
(
	echo ""
	echo 'export PATH=$PATH:'$DIR'/go/bin/'
) >>$PROFILE

source $PROFILE

mkdir -p $GOPATH/src/github.com
