#!/usr/bin/env bash
if [[ -n "$1" ]]
then
  DIR=$1
else
  DIR=$HOME
fi

echo -e "\n\nInstalling Go\n"

mkdir -p $DIR/.go;
cd $DIR/.go;
GOFILE=go1.21.4.darwin-amd64.pkg
wget https://dl.google.com/go/$GOFILE;
tar -xf $GOFILE;
echo "Deleting go in /usr/local"
sudo rm -rf /usr/local/go
sudo mv go /usr/local;
rm $GOFILE;
cd -;

./install/export-go-path-linux.sh $DIR
