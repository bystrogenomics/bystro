FROM perl:5.28.0-slim

ENV PATH="/bystro/bin:/usr/local/go/bin:/go/bin/:${PATH}" \
    PERL5LIB="/bystro/lib:${PERL5LIB}" \
    GOPATH="/go"

COPY . /bystro

WORKDIR /bystro

RUN apt-get update \
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
    &&  npm install -g pm2

RUN git config --global url."https://".insteadOf git://
RUN bash install/install-perl-libs.sh
RUN bash install/install-lmdb-linux.sh
RUN bash install/install-go-linux.sh
RUN bash install/install-go-packages.sh


WORKDIR /bystro/bin
