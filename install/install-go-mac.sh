#!/usr/bin/env bash
if [[ -n "$1" ]]
then
  DIR=$1
else
  DIR=$HOME
fi

echo -e "\n\nInstalling Go\n"

rm -rf go;
echo "Deleting /usr/local/go"
sudo rm -rf /usr/local/go;
rm -f go1.12.9.darwin-amd64.tar.gz;
wget https://dl.google.com/go/go1.12.9.darwin-amd64.tar.gz;
tar -xf go1.12.9.darwin-amd64.tar.gz;
sudo mv go /usr/local;
rm go1.12.9.darwin-amd64.tar.gz;

./install/export-go-path-linux.sh $DIR