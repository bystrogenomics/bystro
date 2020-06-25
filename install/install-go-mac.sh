#!/usr/bin/env bash
if [[ -n "$1" ]]
then
  DIR=$1
else
  DIR=$HOME
fi

echo -e "\n\nInstalling Go\n"

GOVERSION="go1.13.6.darwin-amd64.pkg"
wget https://dl.google.com/go/$GOVERSION;
tar -xf $GOVERSION;
echo "Deleting go in /usr/local"
sudo rm -rf /usr/local/go
sudo mv go /usr/local;
rm $GOVERSION;

./install/export-go-path-linux.sh $DIR
