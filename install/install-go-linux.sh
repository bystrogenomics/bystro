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
rm -rf go
rm go1.11.linux-amd64.tar.gz;
wget https://dl.google.com/go/go1.11.linux-amd64.tar.gz;
tar -xf go1.11.linux-amd64.tar.gz;
echo "Deleting go in /usr/local"
sudo rm -rf /usr/local/go
sudo mv go /usr/local;
rm go1.11.linux-amd64.tar.gz;

. install/export-go-path-linux.sh $DIR $PROFILE