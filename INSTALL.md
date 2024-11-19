# Installing the Bystro Python libraries and cli tools

To install the Bystro Python package, which contains our machine learning library and some application in genetics and proteomics, run:

```sh
pip install bystro
```

The Bystro ancestry CLI `score` tool (`bystro-api ancestry score`) parses VCF files to generate dosage matrices. This requires `bystro-vcf`, a Go program which can be installed with:

```sh
# Requires Go: install from https://golang.org/doc/install
go install github.com/bystrogenomics/bystro-vcf@2.2.3
```

Bystro is compatible with Linux and MacOS. Windows support is experimental. If you are installing on MacOS as a native binary (Apple ARM Architecture), you will need to install the following additional dependencies:

```sh
brew install cmake
```

## Installing the Bystro Annotator

Besides the Bystro ML library, which lives in bystro/python, we also have a Perl library that is used to annotate genetic data, providing necessary information for the Bystro ML library bioinformatics modules.

The Bystro Annotator, which handles processing genetic data (VCF files), performing quality control, feature labeling (annotation) of variants and samples, and generating an annotation output and genotype dosage matrices, is written in Perl.

To install and configure the Bystro Annotator, follow the instructions in [perl/INSTALL.md](perl/INSTALL.md).

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
