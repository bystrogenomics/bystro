#!/usr/bin/env bash
if [[ -n "$1" ]]
then
  DIR=$1
else
  DIR=$HOME
fi

if [[ -n "$2" ]]
then
  PROFILE=$2;
else
  PROFILE=~/.bash_profile;
fi

echo -e "\n\nInstalling Go in /usr/local\n"

GOVERSION="go1.13.6.linux-amd64.tar.gz"
wget https://dl.google.com/go/$GOVERSION;
tar -xf $GOVERSION;
echo "Deleting go in /usr/local"
sudo rm -rf /usr/local/go
sudo mv go /usr/local;
rm $GOVERSION;

. install/export-go-path-linux.sh $DIR $PROFILE
GO111MODULE=on go get github.com/mikefarah/yq/v3
