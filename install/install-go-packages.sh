#!/usr/bin/env bash

echo -e "\n\nInstalling go packages (bystro-vcf, stats, snp)\n"

mkdir -p $GOPATH/src/github.com;

go mod init bystro

go install github.com/akotlar/bystro-stats@1.0.0;

go install github.com/bystrogenomics/bystro-vcf@2.1.1;

go install github.com/akotlar/bystro-snp@1.0.0;

# allows us to modify our config files in place
go install github.com/mikefarah/yq@2.4.1;
