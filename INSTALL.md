# Installing the Bystro Python libraries and cli tools

To install the Bystro Python package, which contains our machine learning library and some application in genetics and proteomics, run:

```sh
pip install --pre bystro
```

Bystro is compatible with Linux and MacOS. Windows support is experimental. If you are installing on MacOS as a native binary (Apple ARM Architecture), you will need to install the following additional dependencies:

```sh
brew install cmake
```

## Setting up the Bystro project for development

If you wish to stand up a local development environment, we recommend using Miniconda to manage Bystro Python dependencies: https://docs.conda.io/projects/miniconda/en/latest/

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
