#!/usr/bin/env bash

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

# Bystro uses LMDB as its db engine. Fast, great use of cache
git clone git://github.com/LMDB/lmdb.git;
make --quiet -C lmdb/libraries/liblmdb;
sudo make --quiet install -C lmdb/libraries/liblmdb;

# Bystro is increasingly a golang progrma. Perl currently handles db fetching,
wget https://storage.googleapis.com/golang/go1.8.linux-amd64.tar.gz;
tar -xf go1.8.linux-amd64.tar.gz;
sudo mv go /usr/local;

# LiftOver is used for the LiftOverCadd.pm package, to liftOver cadd to hg38
# and cadd's GRCh37.p13 MT to hg19
wget http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/liftOver && chmod +x liftOver && sudo mv $_ /usr/local/bin/;

(echo "" ; echo 'export PATH=$PATH:/usr/local/go/bin') >> ~/.bash_profile;
(echo ""; echo 'export GOPATH=$HOME/go') >> ~/.bash_profile;
echo 'export PATH=$PATH:$HOME/go/bin/' >> ~/.bash_profile && source ~/.bash_profile;

mkdir -p $GOPATH/src/github.com;

go get github.com/akotlar/bystro-stats;
go install github.com/akotlar/bystro-stats;

go get github.com/akotlar/bystro-vcf;
go install github.com/akotlar/bystro-vcf;

go get github.com/akotlar/bystro-snp;
go install github.com/akotlar/bystro-snp;

# allows us to modify our config files in place
go get github.com/mikefarah/yaml;
go install github.com/mikefarah/yaml;

rm go1.8*;

# Eases package installation, gives us ability to quickly upgrade perl versions
wget -O - https://install.perlbrew.pl | bash;
(echo "" ; echo "source ~/perl5/perlbrew/etc/bashrc") >> ~/.bash_profile && source ~/.bash_profile;
perlbrew install perl-5.26.0 && perlbrew switch perl-5.26.0;

source ~/.bash_profile;
perl -MCPAN -e 'my $c = "CPAN::HandleConfig"; $c->load(doit => 1, autoconfig => 1); $c->edit(prerequisites_policy => "follow"); $c->edit(build_requires_install_policy => "yes"); $c->commit'

./update-packages.sh;
./install-perl-libs.sh;

printf "\n\nDone, congrats! REMEMBER TO INCREASE ULIMIT ABOVE 1024 IF RUNNING MANY FORKS\n\n";

mkdir -p logs;
