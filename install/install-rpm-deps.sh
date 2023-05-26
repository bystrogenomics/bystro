#!/usr/bin/env bash

echo -e "\n\nInstalling Debian (rpm) dependencies\n";

sudo yum install gcc -y;
sudo yum install openssl -y;
sudo yum install openssl-devel -y;
# Not strictly necessary, useful however for much of what we do
sudo yum install git-all -y;
# pigz for Bystro, used to speed up decompression primarily
sudo yum install pigz -y;
sudo yum install unzip -y;
sudo yum install wget -y;
# For tests involving querying ucsc directly
sudo yum install mysql-devel -y;
# For Search::Elasticsearch::Client::5_0::Direct
sudo yum install libcurl-devel -y

# for perlbrew, in case you want to install a different perl version
#https://www.digitalocean.com/community/tutorials/how-to-install-perlbrew-and-manage-multiple-versions-of-perl-5-on-centos-7
# centos 7 doesn't include bzip2
sudo yum install bzip2  -y;
sudo yum install lz4 --enable-repo=epel;
sudo yum install patch -y;

sudo yum install cpan -y;

curl --silent --location https://rpm.nodesource.com/setup_12.x | sudo bash -;

sudo yum install nodejs -y;

sudo npm install -g pm2;

sudo yum install awscli -y;