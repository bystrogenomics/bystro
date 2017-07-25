#!/usr/bin/env bash

sudo yum install gcc -y
sudo yum install cpan -y
sudo yum install openssl -y
sudo yum install openssl-devel -y
# Not strictly necessary, useful however for much of what we do
sudo yum install git-all -y
# pigz for Bystro, used to speed up decompression primarily
sudo yum install pigz -y
sudo yum install unzip -y
sudo yum install wget -y
# For tests involving querying ucsc directly
sudo yum install mysql-devel -y

# for perlbrew, in case you want to install a different perl version
#https://www.digitalocean.com/community/tutorials/how-to-install-perlbrew-and-manage-multiple-versions-of-perl-5-on-centos-7
# centos 7 doesn't include bzip2
sudo yum install bzip2  -y
sudo yum install patch -y

# Bystro uses LMDB as its db engine. Fast, great use of cache
git clone git://github.com/LMDB/lmdb.git
make -C lmdb/libraries/liblmdb
sudo make install -C lmdb/libraries/liblmdb

# Bystro is increasingly a golang progrma. Perl currently handles db fetching,
wget https://storage.googleapis.com/golang/go1.8.linux-amd64.tar.gz
tar -xvf go1.8.linux-amd64.tar.gz
sudo mv go /usr/local

(echo "" ; echo 'export PATH=$PATH:/usr/local/go/bin') >> ~/.bash_profile 
(echo ""; echo 'export GOPATH=$HOME/go') >> ~/.bash_profile
echo 'export PATH=$PATH:$HOME/go/bin/' >> ~/.bash_profile && source ~/.bash_profile

mkdir -p $GOPATH/src/github.com

go get github.com/akotlar/bystro-stats
go install github.com/akotlar/bystro-stats

go get github.com/akotlar/bystro-vcf
go install github.com/akotlar/bystro-vcf

go get github.com/akotlar/bystro-snp
go install github.com/akotlar/bystro-snp

# Eases package installation, gives us ability to quickly upgrade perl versions
wget -O - https://install.perlbrew.pl | bash
(echo "" ; echo "source ~/perl5/perlbrew/etc/bashrc") >> ~/.bash_profile && source ~/.bash_profile
perlbrew install perl-5.26.0 && perlbrew switch perl-5.26.0

source ~/.bash_profile
perl -MCPAN -e 'my $c = "CPAN::HandleConfig"; $c->load(doit => 1, autoconfig => 1); $c->edit(prerequisites_policy => "follow"); $c->edit(build_requires_install_policy => "yes"); $c->commit'

./install-perl-libs.sh

echo "REMEMBER TO INCREASE ULIMIT ABOVE 1024 IF RUNNING MANY FORKS"

mkdir logs