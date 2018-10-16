FROM perl:latest

ARG v=b10
ENV PATH="/root/bystro/bin:/usr/local/go/bin:/root/go/bin/:${PATH}"
ENV PERL5LIB="/root/perl5/lib/perl5:/root/bystro/lib:${PERL5LIB}"
ENV GOPATH="/root/go"

RUN cpanm --local-lib=/root/perl5 local::lib && eval $(perl -I /root/perl5/lib -Mlocal::lib)
RUN git clone https://github.com/akotlar/bystro
RUN apt-get update && apt-get install sudo

WORKDIR /root/bystro
RUN git checkout ${v}
RUN . install/install-lmdb-linux.sh
RUN wget https://dl.google.com/go/go1.11.linux-amd64.tar.gz \
    && tar -xf go1.11.linux-amd64.tar.gz \
    && mv go /usr/local

RUN . install/install-go-packages.sh
RUN . install/install-perl-libs.sh

WORKDIR /root/bystro/bin