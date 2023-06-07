FROM perl:5.36.1

ENV PATH="/bystro/bin:/usr/local/go/bin:/go/bin/:${PATH}" \
    PERL5LIB="/bystro/lib:${PERL5LIB}" \
    GOPATH="/go"

RUN cpanm --local-lib=/root/perl5 local::lib && eval $(perl -I /root/perl5/lib -Mlocal::lib)
ADD ./ /root/bystro/
RUN apt-get update && apt-get install sudo
RUN git config --global url."https://".insteadOf git://

WORKDIR /root/bystro
RUN . install/install-lmdb-linux.sh
RUN wget https://dl.google.com/go/go1.13.6.linux-amd64.tar.gz \
    && tar -xf go1.13.6.linux-amd64.tar.gz \
    && mv go /usr/local

ADD . /bystro
WORKDIR /bystro

RUN git config --global url."https://".insteadOf git://
RUN bash install/install-go-linux.sh
RUN bash install/install-go-packages.sh
RUN bash install/install-lmdb-linux.sh
RUN bash install/install-perl-libs.sh


WORKDIR /bystro/bin
