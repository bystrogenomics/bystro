FROM perl:5.36.0-slim

ENV PATH="/bystro/bin:/usr/local/go/bin:/go/bin/:${PATH}" \
    PERL5LIB="/bystro/lib:${PERL5LIB}" \
    GOPATH="/go"

RUN apt-get update \
    && apt-get install -y wget sudo gnupg \
    && wget -qO- https://deb.nodesource.com/setup_16.x | bash \
    && apt-get install -y \

    build-essential \
    git \
    openssl libssl-dev \
    pigz \
    unzip \
    wget \
    default-libmysqlclient-dev \
    bzip2 \
    nodejs \
    patch \
    &&  npm install -g pm2

ADD . /bystro
WORKDIR /bystro

RUN git config --global url."https://".insteadOf git://
RUN bash install/install-go-linux.sh
RUN bash install/install-go-packages.sh
RUN bash install/install-lmdb-linux.sh
RUN bash install/install-perl-libs.sh


WORKDIR /bystro/bin
