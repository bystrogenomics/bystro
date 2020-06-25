#!/usr/bin/env bash
if [[ -n "$1" ]]
then
  DIR=$1
else
  DIR=$HOME
fi

echo -e "\n\nInstalling Go\n"

GOFILE=go1.13.5.darwin-amd64.pkg
wget https://dl.google.com/go/$GOFILE;
tar -xf $GOFILE;
echo "Deleting go in /usr/local"
sudo rm -rf /usr/local/go
sudo mv go /usr/local;
rm $GOFILE;

./install/export-go-path-linux.sh $DIR