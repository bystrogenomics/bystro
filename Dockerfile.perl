# Use an official Ubuntu base image
FROM ubuntu:24.04

# Set environment variables
ENV PERL_VERSION=5.34.0

# Set environment variables to reduce image size
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies: sudo
RUN apt update && apt install -y git sudo

# needed for minimal installs for Perl to compile
RUN apt-get update && apt-get install -y --no-install-recommends \
    perl \
    man-db \
    groff \
    libperl-dev

COPY perl /bystro/perl
COPY go /bystro/go
COPY install /bystro/install

# Copy your install-apt.sh script into the container
COPY install-apt.sh /bystro/install-apt.sh

# Install dependencies
RUN cd /bystro && ./install-apt.sh

# Symlink everything in /bystro/perl/bin to /usr/local/bin
ENTRYPOINT ["/bin/bash", "-c", "source ~/.profile && if [ \"$#\" -eq 0 ]; then bystro-annotate.pl --help; else exec \"$@\"; fi", "--"]
