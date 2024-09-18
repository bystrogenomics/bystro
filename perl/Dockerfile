# ---- Build Golang Binaries ----
FROM golang:1.21.3 AS go-builder

# Set the environment variable for Go binaries. This makes sure the binaries are saved to a defined path.
ENV GOBIN=/app/bin

# Install the specific versions of the Go programs
RUN go install github.com/akotlar/bystro-stats@1.0.1
RUN go install github.com/bystrogenomics/bystro-vcf@2.2.3
RUN go install github.com/akotlar/bystro-snp@1.0.1
RUN go install github.com/mikefarah/yq/v2@2.4.1

COPY ./go /app/bystro-go-tools

RUN cd /app/bystro-go-tools && go install bystro/cmd/dosage

# ---- Perl Stage ----
# see https://hub.docker.com/_/perl/
# comes with cpanm
FROM perl:5.34

# Copy the compiled Go binaries from the builder stage
COPY --from=go-builder /app/bin/ /app/bin/

# Add app/bin to PATH
ENV PATH="/app/bin:${PATH}"

# Update package lists and install dependencies
# Note: installing libmariadb-dev rather than libmysql-dev since the latter was removed
RUN apt-get update \
  && apt-get install -y \
  build-essential \
  bzip2 \
  git \
  libcurl4-openssl-dev \
  liblmdb-dev \
  libmariadb-dev \
  libssl-dev \
  lz4 \
  openssl \
  patch \
  pigz \
  pkg-config \
  unzip \
  wget \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app/liftOver

# Install liftOver into a common system directory, /usr/local/bin
RUN wget --quiet http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/liftOver \
  && chmod +x liftOver \
  && mv liftOver /usr/local/bin

RUN rm -rf /app/liftOver

# Change working directory to perl-deps
WORKDIR /app/perl-deps

# install cpm for faster installation of perl packages
RUN curl -fsSL https://raw.githubusercontent.com/skaji/cpm/main/cpm | perl - install -g App::cpm

# Install Dist::Zilla
RUN cpm install -g Dist::Zilla Archive::Tar::Wrapper

# Install one-off Bystro dependencies that require special installation
RUN cpm install -g https://github.com/bystrogenomics/msgpack-perl.git
RUN cpm install -g --no-test MouseX::Getopt

# Remove perl-deps directory
RUN rm -rf /app/perl-deps

# Setup bystro-perl directory to build and install distribution
WORKDIR /app/bystro-perl

# Copy Bystro Package
COPY . /app/bystro-perl/

# Install build dependencies
RUN cpm install -g --show-build-log-on-failure $(dzil authordeps --missing)

# install Bystro package dependencies
RUN cpm install -g --show-build-log-on-failure $(dzil listdeps --missing)

# Test and Install Bystro
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
