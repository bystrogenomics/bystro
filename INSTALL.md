# Table of Contents

1. [Installing Bystro Using Docker](#installing-bystro-using-docker)
   - [Building the Latest Version of Bystro in Docker](#building-the-latest-version-of-bystro-in-docker)
2. [Direct (Non-Docker) Installation](#direct-non-docker-installation)
   - [Installing the Bystro Annotator (Perl/CLI)](#installing-the-bystro-annotator-perlcli)
     - [For RPM-based Distros (Fedora, Red Hat, CentOS, etc.)](#fedora-redhat-centos-opensuse-mandriva)
     - [For MacOS (Tested on High Sierra)](#macos-tested-on-highsierra-interactive)
     - [For Ubuntu](#ubuntu)
   - [Installing Bystro Python Libraries and CLI Tools](#installing-the-bystro-python-libraries-and-cli-tools)
3. [Configuring the Bystro Annotator](#configuring-the-bystro-annotator)
4. [Databases](#databases)
5. [Running Your First Annotation](#running-your-first-annotation)

For most users, we recommend not installing the software, and using https://bystro.io, where the software is hosted

The web app provides full functionality for any size experiment, a convenient search interface, and excellent performance

# Installing Bystro using Docker

###### The recommended way to use Bystro on the command line

Make sure you have [Docker installed](https://store.docker.com/search?type=edition&offering=community)

#### Building the latest version of Bystro in Docker

```
git clone https://github.com/bystrogenomics/bystro.git && cd bystro
docker build -t bystro .
docker run bystro bystro-annotate.pl #Annotate
docker run bystro bystro-build.pl #Build
```

# Direct (non-Docker) installation

There are 2 components to Bystro:
 1. The Bystro annotator: a Perl program accessed through the command line (via bin/bystro-*)
 2. The Bystro Python package: where the rest of Bystro's functionality lives (statistics, proteomics, etc).

## Installing the Bystro annotator (Perl/cli)

##### (Fedora, Redhat, Centos, openSUSE, Mandriva)

1.  `git clone https://github.com/bystrogenomics/bystro.git && cd bystro && source ./install-rpm.sh`

##### MacOS (tested on HighSierra, interactive)

1.  `git clone https://github.com/bystrogenomics/bystro.git && cd bystro && source ./install-mac.sh`

##### Ubuntu
1.  Ensure that packages are up to date (`sudo apt update`), or that you are satisified with the state of package versions.
2.  `git clone https://github.com/bystrogenomics/bystro.git && cd bystro && source ./install-apt.sh`
    - Please not that this installation script will ask you for the root password in order to install system dependencies

## Installing the Bystro Python libraries and cli tools

We recommend using Miniconda to manage Bystro Python dependencies: https://docs.conda.io/projects/miniconda/en/latest/

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


The easiest way to get started:
```sh
# Installs the Bystro Python library and cli, and starts the Ray cluster
# alternatively you could have run `make install && make ray-start-local`
# or just `make install` if you only wished to install Bystro
make run-local
```
- This will create a local Ray server, which is needed for some Bystro operations
- To stop Ray: `ray stop`


If you are developing/contributing to the Bystro library, for a faster build use:
```sh
make develop
```

To start a local beanstalkd listener, use either `make serve-local` or `make serve-dev`, depending on whether you are in a product environment, or development.

Follow the instructions below to install the Bystro annotator:

## Configuring the Bystro annotator

Once Bystro is installed, it needs to be configured. The easiest step is choosing the species/assemblies to annotate.

1. Download the Bystro database for your species/assembly

- **Example:** hg38 (human reference GRCh38): `wget https://s3.amazonaws.com/bystro-db/hg38_v7.tar.gz`</strong>
  - You need ~700GB of free space for hg38 and  ~400GB of free space for hg19, including the space for the tar.gz archives

2. To install the database:

   **Example:**

   ```shell
   cd /mnt/annotator/
   wget https://s3.amazonaws.com/bystro-db/hg38_v7.tar.gz
   bgzip -d -c --threads 32 hg38_v7.tar.gz | tar xvf -
   ```

   In this example the hg38 database would located in `/mnt/annotator/hg38`

3. Update the YAML configuration for the species/assembly to point to the database.

   For human genome assemblies, we provide pre-configured hg19.yml and hg38.yml, which assume `/mnt/annotator/hg19_v9` and `/mnt/annotator/hg38_v7` database directories respectively.

   If using a different mount point, different database folder name, or a different (or custom-built) database altogether,
   you will need to update the `database_dir` property of the yaml config.
     - Note for a custom database, you would also need to ensure the track `outputOrder` lists all tracks, and that each track has all desired `features` listed

   For instance, using `yq` to can configure the `database_dir` and set `temp_dir` to have in-progress annotations written to local disk

   ```shell
   yq write -i config/hg38.yml database_dir /mnt/my_fast_local_storage/hg38_v7
   yq write -i config/hg38.yml temp_dir /mnt/my_fast_local_storage/tmp
   ```

## Databases:

1. Human (hg38): https://s3.amazonaws.com/bystro-db/hg38_v7.tar.gz
2. Human (hg19): https://s3.amazonaws.com/bystro-db/hg19_v9.tar.gz
3. There are no restrictions on species support, but we currently only build human genomes. Please create a GitHub issue if you would like us to support others.

## Running your first annotation

Ex: Runing hg38 annotation

```sh
bin/bystro-annotate.pl --config config/hg38.yml --in /path/in.vcf.gz --out /path/outPrefix --run_statistics [0,1] --compress
```

The outputs will be:

- Annotation (compressed, due to --compress flag): `outPrefix.annotation.tsv.gz`
- Annotation log: `outPrefix.log.txt`
- Statistics JSON file `outPrefix.statistics.json`
- Statistics tab-separated file: `outPrefix.statistics.tsv`
  - Removing the `--run_statistics` flag will skip the generation of `outPrefix.statistics.*` files
