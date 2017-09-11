#!/bin/sh
cdir=$(pwd)
line=$(lsblk | grep nvme | wc -l);

cd ~/seq-lmdb

# Update go packages;
# TODO: combine go packages
cd ~/go/src/github.com/akotlar/bystro-vcf/
git pull origin master 
cd ../bystro-snp
git pull origin master
cd ../bystro-utils
git pull origin master

sudo mkdir -p /mnt/annotator;
sudo chown ec2-user -R /mnt/annotator;

if (($line >= 2)); then 
  sudo mdadm --create --verbose /dev/md0 --level=0 --name=ANNOTATOR --raid-devices=2 /dev/nvme0n1 /dev/nvme1n1;
  sudo mkfs.ext4 -L ANNOTATOR /dev/md0;
  sudo mount LABEL=ANNOTATOR /mnt/annotator;
else 
  sudo mkfs.ext4 -L ANNOTATOR /dev/nvme0n1;
  sudo mount LABEL=ANNOTATOR /mnt/annotator;
fi

echo "LABEL=ANNOTATOR       /mnt/annotator   ext4    defaults,nofail        0       2" | sudo tee -a /etc/fstab;

aws s3 cp s3://bystro-db /mnt/annotator/ --recursive --include "*.tar.gz"

# Copy latest database files, and untar them
cd /mnt/annotator

for db in *.tar.gz
do
  echo "Working on $db";
  pigz -d -c $db | tar xvf -
done

cd $cdir