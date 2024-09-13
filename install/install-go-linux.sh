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

# Clean in case somethign left over from old installation
mkdir -p $DIR/.go;
cd $DIR/.go;
GOFILE=go1.21.4.linux-amd64.tar.gz
wget https://dl.google.com/go/$GOFILE;
tar -xf $GOFILE;
echo "Deleting go in /usr/local"
sudo rm -rf /usr/local/go
sudo mv go /usr/local;
rm $GOFILE;
cd -;

. install/export-go-path-linux.sh $DIR $PROFILE
