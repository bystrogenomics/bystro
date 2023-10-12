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

# Copy the compiled Go binaries from the builder stage
COPY --from=go-builder /app/bin/ /app/bin/

# Add app/bin to PATH
ENV PATH="/app/bin:${PATH}"

# Update package lists and install dependencies
# Note: installing libmariadb-dev rather than libmysql-dev since the latter was removed
RUN apt-get update \
  && apt-get install -y \
  awscli \
  build-essential \
  bzip2 \
  git \
  libcurl4-openssl-dev \
  liblmdb-dev \
  libmariadb-dev \
  libssl-dev \
  lz4 \
  nodejs \
  npm \
  openssl \
  patch \
  pigz \
  pkg-config \
  unzip \
  wget \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Install pm2
RUN npm install -g pm2

WORKDIR /app/liftOver

# Install liftOver into a common system directory, /usr/local/bin
RUN wget --quiet http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/liftOver \
  && chmod +x liftOver \
  && mv liftOver /usr/local/bin

RUN rm -rf /app/liftOver

# Change working directory to perl-deps
WORKDIR /app/perl-deps

# Install Dist::Zilla
RUN cpanm --quiet Dist::Zilla Archive::Tar::Wrapper

# Install one-off Bystro dependencies that require special installation
RUN cpanm --quiet https://github.com/bystrogenomics/msgpack-perl.git
RUN cpanm --quiet --notest MouseX::Getopt
RUN git clone --depth 1 --recurse-submodules https://github.com/salortiz/LMDB_File.git \
  && cd LMDB_File \
  && cpanm --quiet .

# Install workaround because default DBD::mysql, i.e. version >5.x, will not automatically install on ubuntu
RUN cpanm --quiet DBD::mysql@4.051

# Remove perl-deps directory
RUN rm -rf /app/perl-deps

# Setup bystro-perl directory to build and install distribution
WORKDIR /app/bystro-perl

# Copy Bystro Package
COPY . /app/bystro-perl/

# Install build dependencies
RUN dzil authordeps --missing | cpanm --quiet

# install Bystro package dependencies
RUN dzil listdeps --missing | cpanm --quiet

# Build Bystro package and
RUN dzil install

RUN rm -rf /app/perl-deps

WORKDIR /app

# Copy entry point script that launches Bystro perl scripts
COPY entrypoint.sh /app/bin/

# Copy perl scripts
COPY ./bin/ /app/bin/

# Remove perl installation
RUN rm -rf /app/bystro-perl

RUN chmod +x /app/bin/entrypoint.sh

ENTRYPOINT ["/app/bin/entrypoint.sh"]
