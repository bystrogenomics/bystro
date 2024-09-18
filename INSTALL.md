# Installing the Bystro Python libraries and cli tools

To install the Bystro Python package, which contains our machine learning library and some application in genetics and proteomics, run:

```sh
pip install --pre bystro
```

The Bystro ancestry CLI `score` tool (`bystro-api ancestry score`) parses VCF files to generate dosage matrices. This requires `bystro-vcf`, a Go program which can be installed with:

```sh
# Requires Go: install from https://golang.org/doc/install
go install github.com/bystrogenomics/bystro-vcf@2.2.2
```

Bystro is compatible with Linux and MacOS. Windows support is experimental. If you are installing on MacOS as a native binary (Apple ARM Architecture), you will need to install the following additional dependencies:

```sh
brew install cmake
```

## Setting up the Bystro project for development

If you wish to stand up a local development environment, we recommend using Miniconda to manage Bystro Python dependencies: https://docs.conda.io/projects/miniconda/en/latest/

Once Bystro annotator installation is complete, and assuming Conda/Miniconda has been installed, run :

```sh
# Install Rust
curl https://sh.rustup.rs -sSf | sh -s -- -y
echo -e "\n### Bystro: Done installing Rust! Now sourcing .cargo/env for use in the current shell ###\n"
source "$HOME/.cargo/env"
# Create or activate Bystro conda environment and install all dependencies
# This assumes you are in the bystro folder
source .initialize_conda_env.sh;
```

If you edit Cython or Rust code, you will need to recompile the code. To do this, run:

```sh
make build-python-dev
```

## Starting Bystro API server listeners

If you have a private deployment of the Bystro cluster, you will need to start the API server listeners.

To start local workers for receiving and processing jobs from the Bystro API server, use either `make serve-local` or `make serve-dev`, depending on whether you are in a product environment, or development (the difference is entirely in build time and optimization flags).

These workers rely on a few Go programs, which are compiled and installed in the `go/bin` directory. If you are running the workers for the first time, [you will need to install Go](https://go.dev/dl/), and then install the Go programs by running:

```sh
make install-go
```

To start the workers, update `config/beanstalk.yml` to point to the correct beanstalk servers (the ones that the Bystro API server is pointing to), and `config/opensearch.yml` to point to an OpenSearch server, and then run:

```sh
make serve-dev
```

## Installing the Bystro Annotator

Besides the Bystro ML library, which lives in bystro/python, we also have a Perl library that is used to annotate genetic data, providing necessary information for the Bystro ML library bioinformatics modules.

The Bystro Annotator, which handles processing genetic data (VCF files), performing quality control, feature labeling (annotation) of variants and samples, and generating an annotation output and genotype dosage matrices, is written in Perl.

To install and configure the Bystro Annotator, follow the instructions in [perl/INSTALL.md](perl/INSTALL.md).
