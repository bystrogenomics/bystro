# Use an official Ubuntu base image
FROM ubuntu:24.04

# Set environment variables
ENV PERL_VERSION=5.34.0

# Install dependencies: sudo
RUN apt update && apt install -y git sudo
# needed for minimal installs for Perl to compile
RUN apt install -y unminimize && yes | unminimize

COPY perl /bystro/perl
COPY go /bystro/go
COPY install /bystro/install

# Copy your install-apt.sh script into the container
COPY install-apt.sh /bystro/install-apt.sh

# Install dependencies
RUN cd /bystro && ./install-apt.sh

# Ensure .profile gets sourced when the container starts
CMD ["/bin/bash", "-c", "source ~/.profile && exec bash"]