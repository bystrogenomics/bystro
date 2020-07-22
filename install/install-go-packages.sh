#!/usr/bin/env bash

echo -e "\n\nInstalling go packages (bystro-vcf, stats, snp)\n"

mkdir -p $GOPATH/src/github.com;

GO111MODULE=on go get github.com/akotlar/bystro-stats@1.0.0;

GO111MODULE=on go get github.com/akotlar/bystro-vcf@1.0.0;

GO111MODULE=on go get github.com/akotlar/bystro-snp@1.0.0;

# allows us to modify our config files in place
GO111MODULE=on go get github.com/mikefarah/yq/v3;
