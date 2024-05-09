# Bystro Annotator Installation Guide

### Table of Contents
1. [Installing the Bystro Perl Annotator](#installing-the-bystro-perl-annotator)
2. [Installing Bystro Python Libraries](#installing-bystro-python-libraries)
3. [Configuring the Bystro Annotator](#configuring-the-bystro-annotator)
4. [Databases](#databases)
5. [Running Your First Annotation](#running-your-first-annotation)
6. [FAQ](#faq)

## Installing the Bystro Perl Annotator
#### Amazon 2023

1. Clone and install the Bystro repository:
    ```sh
    git clone https://github.com/bystrogenomics/bystro.git && cd bystro && source ./install-rpm.sh
    ```

2. Install dependencies:
    ```sh
    cpanm --quiet https://github.com/bystrogenomics/msgpack-perl.git
    cpanm --quiet --notest MouseX::Getopt
    git clone --depth 1 --recurse-submodules https://github.com/salortiz/LMDB_File.git \
      && cd LMDB_File \
      && cpanm --quiet . \
      && cd .. \
      && rm -rf LMDB_File
    cpanm --quiet DBD::mysql@4.051
    # Ensure mysql-devel is installed as it's needed for mysql_config
    ```

## Installing Bystro Python Libraries
We recommend using Miniconda to manage Python dependencies for Bystro. After installing Conda, proceed with the following steps:

1. Install Rust:
    ```sh
    curl https://sh.rustup.rs -sSf | sh -s -- -y
    echo -e "\n### Bystro: Done installing Rust! Now sourcing .cargo/env for use in the current shell ###\n"
    source "$HOME/.cargo/env"
    ```

2. Set up the Bystro environment:
    ```sh
    source .initialize_conda_env.sh
    ```

3. To manage local operations:
    ```sh
    make run-local # Starts a local Ray server
    ray stop # To stop Ray
    make develop # For faster builds during development
    make serve-local # Start a local beanstalkd listener in a product environment
    make serve-dev # Start in a development environment
    ```

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


## FAQ

This section outlines common issues encountered during the deployment of Bystro in Amazon 2023 and the solutions that addressed these issues at the time.

- **Perl Version Requirement**
  - In order to install Lmbd_file, Perl 5.34.0+ must be installed.

- **Database Configuration Error**
  - **Error:** [fatal] dbCleanUp LMDB error: 13 at /home/ec2-user/bystro/perl/lib/Seq/DBManager.pm line 1177.
  - **Cause:** The `data_dir` in `config/hg19.yml` and `config/hg38.yml` might not be configured correctly, or the hg19 and hg38 databases were not downloaded and extracted.
  - **Reference:** [Configuring the Bystro Annotator](#configuring-the-bystro-annotator).

- **Missing mysql_config**
  - **Error:** Can't exec "mysql_config": No such file or directory at Makefile.PL line 89. Cannot find the file 'mysql_config'! Your execution PATH doesn't seem to contain the path to mysql_config.
  - **Solution:**
    ```sh
    sudo wget https://dev.mysql.com/get/mysql80-community-release-el9-1.noarch.rpm
    sudo dnf install mysql80-community-release-el9-1.noarch.rpm -y
    sudo rpm --import https://repo.mysql.com/RPM-GPG-KEY-mysql-2023
    sudo rm mysql80-community-release-el9-1.noarch.rpm
    sudo yum install mysql-devel -y;
    ```

- **PERLIO::gzip Installation Failure**
  - **Error:**
    ```
    PerlIO/gzip/gzip.bs 644
    "/home/ec2-user/perl5/perlbrew/perls/perl-5.34.0/bin/perl" "/home/ec2-user/perl5/perlbrew/perls/perl-5.34.0/lib/5.34.0/ExtUtils/xsubpp" -typemap '/home/ec2-user/perl5/perlbrew/perls/perl-5.34.0/lib/5.34.0/ExtUtils/typemap' gzip.xs > gzip.xsc
    mv gzip.xsc gzip.c
    cc -c -fwrapv -fno-strict-aliasing -pipe -fstack-protector-strong -I/usr/local/include -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 -D_FORTIFY_SOURCE=2 -O2 -DVERSION="0.20" -DXS_VERSION="0.20" -fPIC "-I/home/ec2-user/perl5/perlbrew/perls/perl-5.34.0/lib/5.34.0/x86_64-linux/CORE" gzip.c
    gzip.xs:16:10: fatal error: zlib.h: No such file or directory
    16 | #include <zlib.h>
    |          ^~~~~~~~
    compilation terminated.
    make: *** [Makefile:338: gzip.o] Error 1
    ```
  - **Solution:** Install zlib-devel to resolve this and other related installation issues:
    ```sh
    sudo yum install zlib-devel
    ```