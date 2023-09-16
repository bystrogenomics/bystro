# ---- Build Golang Binaries ----
FROM golang:1.17 AS go-builder

# Set the environment variable for Go binaries. This makes sure the binaries are saved to a defined path.
ENV GOBIN=/app/bin

# Install the specific versions of the Go programs
RUN go install github.com/akotlar/bystro-stats@1.0.0
RUN go install github.com/akotlar/bystro-vcf@1.0.0
RUN go install github.com/akotlar/bystro-snp@1.0.0
RUN go install github.com/mikefarah/yq@2.4.1

# ---- Perl Stage ----
# see https://hub.docker.com/_/perl/
# comes with cpanm
FROM perl:5.38

# Update package lists and install dependencies
# Note: installing libmariadb-dev rather than libmysql-dev since the latter was removed
RUN apt-get update \
    && apt-get install -y \
    build-essential pkg-config git \
    pigz unzip wget openssl \
    bzip2 lz4 patch nodejs npm awscli \
    libcurl4-openssl-dev libssl-dev liblmdb-dev libmariadb-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pm2
RUN npm install -g pm2

# Install liftOver into a common system directory, /usr/local/bin
RUN wget http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/liftOver \
    && chmod +x liftOver \
    && mv liftOver /usr/local/bin

# Set working directory inside the container
WORKDIR /app

# Copy the compiled Go binaries from the builder stage
COPY --from=go-builder /app/bin/ /app/bin/

# Add the golang binary locations to PATH
ENV PATH="/app/bin:${PATH}"

# Install perl deps
RUN cpanm https://github.com/bystrogenomics/msgpack-perl.git
RUN cpanm --force MouseX::Getopt
RUN git clone --recurse-submodules https://github.com/salortiz/LMDB_File.git \
    && cd LMDB_File \
    && perl Makefile.PL \
    && make test \
    && make install

RUN rm -rf /app/LMDB_File

# Copy Bystro Package
COPY ./Bytro-0.001.tar.gz .

# Install Bystro
RUN cpanm -n Bytro-0.001.tar.gz && rm -rf Bytro-0.001.tar.gz

# Setup enyry point that launches individual perl scripts
COPY entrypoint.sh /app/bin/

RUN chmod +x /app/bin/entrypoint.sh

ENTRYPOINT ["/app/bin/entrypoint.sh"]

# FROM perl:5.36.1

# ENV PATH="/bystro/bin:/usr/local/go/bin:/go/bin/:${PATH}" \
#     PERL5LIB="/bystro/lib:${PERL5LIB}" \
#     GOPATH="/go"

# RUN cpanm --local-lib=/root/perl5 local::lib && eval $(perl -I /root/perl5/lib -Mlocal::lib)
# ADD ./ /root/bystro/
# RUN apt-get update && apt-get install sudo
# RUN git config --global url."https://".insteadOf git://

# WORKDIR /root/bystro
# RUN . install/install-lmdb-linux.sh
# RUN wget https://dl.google.com/go/go1.13.6.linux-amd64.tar.gz \
#     && tar -xf go1.13.6.linux-amd64.tar.gz \
#     && mv go /usr/local

# ADD . /bystro
# WORKDIR /bystro

# RUN git config --global url."https://".insteadOf git://
# RUN bash install/install-go-linux.sh
# RUN bash install/install-go-packages.sh
# RUN bash install/install-lmdb-linux.sh
# RUN bash install/install-perl-libs.sh

# WORKDIR /bystro/bin
