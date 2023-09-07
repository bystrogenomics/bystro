For most users, we recommend not installing the software, and using https://bystro.io, where the software is hosted

The web app provides full functionality for any size experiment, a convenient search interface, and excellent performance

# Installing Bystro using Docker

###### The recommended way to use Bystro on the command line

Make sure you have [Docker installed](https://store.docker.com/search?type=edition&offering=community)

#### Building the latest version of Bystro in Docker

```
git clone https://github.com/akotlar/bystro.git && cd bystro
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

1.  `git clone https://github.com/akotlar/bystro.git && cd bystro && source ./install-rpm.sh`

##### MacOS (tested on HighSierra, interactive)

1.  `git clone https://github.com/akotlar/bystro.git && cd bystro && source ./install-mac.sh`

##### Ubuntu
1.  Ensure that packages are up to date (`sudo apt update`), or that you are satisified with the state of package versions.
2.  `git clone https://github.com/akotlar/bystro.git && cd bystro && source ./install-apt.sh`
    - Please note that this installation script requires root priveleges, in order to install system dependencies

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

to install the Python package dependencies. Then, run:
```
# Build the Python package for local use
make build
```

to intall the Bystro Python library.

Follow the instructions below to install the Bystro annotator:

## Configuring the Bystro annotator

Once Bystro is installed, it needs to be configured. The easiest step is choosing the species/assemblies to annotate.

1. Download the Bystro database for your species/assembly

- **Example:** hg38 (human reference GRCh38): `wget https://s3.amazonaws.com/bystro-db/hg38.tar.gz`</strong>
  - You need ~270GB of free space for human and mouse databases (less for all other species by at least half)
  - Databases are ~4.5x their compressed size when unpacked

The downloaded databases are tar balls, in which the database files that Bystro needs to access are located in `<assembly>/index`

2. Expand within some folder:

   **Example:**

   ```shell
   pigz -d -c hg38.tar.gz | (cd /where/to/ && tar xvf -)
   ```

   In this example the hg38 database would located in `/where/to/hg38/index`

3. Create a copy of the assembly config YAML file, and update its `database_dir` property to point to `index` folder

   - Since in our downloaded tarballs, the database is within a folder called `index`

   **Example:**

   ```shell
   cp config/hg38.clean.yml config/hg38.yml;
   yq write -i config/hg38.yml database_dir /mnt/annotator/hg38/index
   yq write -i config/hg38.yml temp_dir /mnt/annotator/tmp
   ```

   The config file editing could of course be also done using vim/nano/vi/emacs

## Databases:

1. Human (hg38): https://s3.amazonaws.com/bystro-db/hg38.tar.gz
2. Human (hg19): https://s3.amazonaws.com/bystro-db/hg19.tar.gz
3. Mouse (mm10): https://s3.amazonaws.com/bystro-db/mm10.tar.gz
4. Mouse (mm9): https://s3.amazonaws.com/bystro-db/mm9.tar.gz
5. Fly (dm6): https://s3.amazonaws.com/bystro-db/dm6.tar.gz
6. Rat (rn6): https://s3.amazonaws.com/bystro-db/rn6.tar.gz
7. Rhesus (rheMac8): https://s3.amazonaws.com/bystro-db/rheMac8.tar.gz
8. S.cerevisae (sacCer3): https://s3.amazonaws.com/bystro-db/sacCer3.tar.gz
9. C.elegans (ce11): https://s3.amazonaws.com/bystro-db/ce11.tar.gz

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
