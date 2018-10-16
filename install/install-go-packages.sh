#!/usr/bin/env bash

echo -e "\n\nInstalling go packages (bystro-vcf, stats, snp)\n"

mkdir -p $GOPATH/src/github.com;

go get github.com/akotlar/bystro-stats;

go get github.com/akotlar/bystro-vcf;

go get github.com/akotlar/bystro-snp;

# allows us to modify our config files in place
go get github.com/mikefarah/yq;
