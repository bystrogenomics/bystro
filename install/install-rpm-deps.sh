#!/usr/bin/env bash

echo "Installing Debian (rpm) dependencies"

sudo yum install gcc -y -q;
sudo yum install cpan -y -q;
sudo yum install openssl -y -q;
sudo yum install openssl-devel -y -q;
# Not strictly necessary, useful however for much of what we do
sudo yum install git-all -y -q;
# pigz for Bystro, used to speed up decompression primarily
sudo yum install pigz -y -q;
sudo yum install unzip -y -q;
sudo yum install wget -y -q;
# For tests involving querying ucsc directly
sudo yum install mysql-devel -y -q;

# for perlbrew, in case you want to install a different perl version
#https://www.digitalocean.com/community/tutorials/how-to-install-perlbrew-and-manage-multiple-versions-of-perl-5-on-centos-7
# centos 7 doesn't include bzip2
sudo yum install bzip2  -y -q;
sudo yum install patch -y -q;