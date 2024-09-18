# Bystro Annotator Package Installation

## Installing Bystro Annotator using Docker

To build a Docker image using the `Dockerfile`, run the following:

```bash
cd ../ && docker build -t bystro-annotator -f Dockerfile.perl .
```

## Installing Bystro Annotator Bare Metal / Locally

The easiest way to install Bystro directly on your machine is to run:

- Debian/Ubuntu: `../install-apt.sh`
- Centos/Fedora/Amazon Linux: `../install-rpm.sh`

You will be prompted for "sudo" access to install the necessary system level dependencies.

## Manual Install

First you'll need to install some prerequisites:

- Debian/Ubuntu: 'sudo ../install/install-apt-deps.sh`
- Centos/Fedora/Amazon Linux: 'sudo ../install/install-rmp-deps.sh`
- bgzip: `../install/install-htslib.sh ~/.profile ~/.local`

Bystro also relies on a few `Go` programs, which can be installed with the following:

```bash
mkdir ~/.local

# Assuming we are installing this on linux, on an x86 processor
# and that our login shell environment is stored in ~/.profile (another common one is ~/.bash_profile)
../install/install-go.sh ~/.profile ~/ ~/.local ~/bystro/ linux-amd64 1.21.4

source ~/.profile
```

The instructions for installing the Bystro Perl library use [`cpm`](https://metacpan.org/pod/App::cpanminus).

- Alternatively you can use [cpanm](https://metacpan.org/dist/App-cpanminus/view/bin/cpanm), which can be installed with the following: `curl -fsSL https://cpanmin.us | perl - App::cpanminus`
- Just replace every `cpm install --test` and `cpm install` command with `cpanm`

To install `cpm`, run the following:

```bash
curl -fsSL https://raw.githubusercontent.com/skaji/cpm/main/cpm | perl - install App::cpm
```

- Note that this will by default have all perl libraries install in `./local`; assuming you are running these instructions from `~/bystro/perl` that means that installed Perl binaries and libraries will be in `~/bystro/perl/local`.
- You will also need to make sure that you have the Bystro libraries and binaries in your PERL5LIB and PATH environment variables. You can do this by adding the following to your `~/.profile` or `~/.bash_profile`:

```bash
# Add this to your ~/.profile or ~/.bash_profile
export PERL5LIB=~/bystro/perl/local/lib/perl5/:~/bystro/perl/lib:$PERL5LIB
export PATH=~/bystro/perl/bin:~/bystro/perl/local/bin:$PATH
```

- Alternatively to have `cpm` install libraries in your `@INC` path, you can run `cpm install -g` instead of `cpm install` (and then you can remove `/bystro/perl/local/lib/perl5/` from your `PERL5LIB` and `/bystro/perl/local/bin` from your `PATH`)

<br>

A few dependencies must be specially separately installed:

```bash
cpm install --test https://github.com/bystrogenomics/msgpack-perl.git

ALIEN_INSTALL_TYPE=share cpm install --test Alien::LMDB
cpm install --test LMDB_File

# no --test option because it has a trivial failure related to formatting of cli help strings
cpm install MouseX::Getopt
```

- Please note, that if you are using Perl > 5.36.0, you will need to manually install LMDB_File 0.14, which will require `make`

  ```bash
  ALIEN_INSTALL_TYPE=share cpm install --test Alien::LMDB
  git clone --depth 1 --recurse-submodules https://github.com/salortiz/LMDB_File.git \
    && cd LMDB_File \
    && git checkout 34acb71d7d86575fe7abb3f7ad95e8653019b282 \
    && perl Makefile.PL && make distmeta \
    && ln -s MYMETA.json META.json && ln -s MYMETA.yml META.yml \
    && cpm install --show-build-log-on-failure --test . \
    && cd ..
    && rm -rf LMDB_File
  ```

Now you can install the rest of the dependencies:

```bash
  cpm install
```

Now you're ready to try Bystro:

```bash
# First let's run our test suite
cd ~/bystro/perl
prove -r ./t -j$(nproc)

# Then let's try running bystro-annotate.pl
bystro-annotate.pl --help
```

## Configuring the Bystro Annotator

Once Bystro is installed, it needs to be configured. The easiest step is choosing the species/assemblies to annotate.

1. Download the Bystro database for your species/assembly

- **Example:** hg38 (human reference GRCh38): `wget https://s3.amazonaws.com/bystro-db/hg38_v11.tar.gz`</strong>
  - You need ~691GB of free space for hg38 and ~376GB of free space for hg19, including the space for the tar.gz archives
    - The unpacked databases are ~517GB for hg38 and ~283GB for hg19

2. To install the database:

   **Example:**

   ```shell
   cd /mnt/annotator/
   wget https://s3.amazonaws.com/bystro-db/hg38_v11.tar.gz
   bgzip -d -c --threads 32 hg38_v11.tar.gz | tar xvf -
   ```

   In this example the hg38 database would located in `/mnt/annotator/hg38`

3. Update the YAML configuration for the species/assembly to point to the database.

   For human genome assemblies, we provide pre-configured hg19.yml and hg38.yml, which assume `/mnt/annotator/hg19_v10` and `/mnt/annotator/hg38_v11` database directories respectively.

   If using a different mount point, different database folder name, or a different (or custom-built) database altogether,
   you will need to update the `database_dir` property of the yaml config.

   - Note for a custom database, you would also need to ensure the track `outputOrder` lists all tracks, and that each track has all desired `features` listed

   For instance, using `yq` to can configure the `database_dir` and set `temp_dir` to have in-progress annotations written to local disk

   ```shell
   yq write -i config/hg38.yml database_dir /mnt/my_fast_local_storage/hg38_v11
   yq write -i config/hg38.yml temp_dir /mnt/my_fast_local_storage/tmp
   ```

## Databases:

1. Human (hg38): https://s3.amazonaws.com/bystro-db/hg38_v11.tar.gz
2. Human (hg19): https://s3.amazonaws.com/bystro-db/hg19_v10.tar.gz
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

## Coding style and tidying

The `.perltidyrc` gives the coding style and `tidyall` from [Code::TidyAll](https://metacpan.org/dist/Code-TidyAll) can be used to tidy all files with `tidyall -a`.
Please tidy all files before submitting patches.

Install tidyall, perltidy, and perlcritic like so

```bash
cpanm Code::TidyAll Perl::Tidy Perl::Critic
```
