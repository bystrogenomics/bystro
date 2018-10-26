FROM perl:5.28.0-slim

ENV PATH="/root/bystro/bin:/usr/local/go/bin:/root/go/bin/:${PATH}" \
    PERL5LIB="/root/perl5/lib/perl5:/root/bystro/lib:${PERL5LIB}" \
    GOPATH="/root/go"

ADD ./ /root/bystro/

WORKDIR /root/bystro

RUN cpanm --local-lib=/root/perl5 local::lib && eval $(perl -I /root/perl5/lib -Mlocal::lib) \
    && apt-get update \
    && apt-get install -y wget sudo gnupg \
    && wget -qO- https://deb.nodesource.com/setup_8.x | bash \
    && apt-get install -y \
    build-essential \
    git \
    openssl libssl-dev \
    pigz \
    unzip \
    wget \
    default-libmysqlclient-dev \
    bzip2 \
    patch \
    nodejs \
    npm \
    &&  npm install -g pm2 \
    && . install/install-lmdb-linux.sh \
    && wget https://dl.google.com/go/go1.11.linux-amd64.tar.gz \
    && tar -xf go1.11.linux-amd64.tar.gz \
    && rm go1.11.linux-amd64.tar.gz
    && mv go /usr/local \
    && . install/install-go-packages.sh \
    && . install/install-perl-libs.sh

WORKDIR /root/bystro/bin
