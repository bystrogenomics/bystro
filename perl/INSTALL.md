# Bystro Annotator Package Installation and Configuration

## Installation

These instructions assume that you are in the `perl` directory of the Bystro repository, e.g. `~/bystro/perl`.

### Installing Bystro Annotator using Docker

To build a Docker image using the `Dockerfile`, run the following:

```bash
cd ../ && docker build -t bystro-annotator -f Dockerfile.perl .
# Run Bystro Annotator from the new Docker container; replace <command> with the desired command
# If no command provided, will automatically run bystro-annotate.pl --help
docker run bystro-annotator <command>
```

- Commands:
  - Run the annotator: `docker run bystro-annotator bystro-annotate.pl --help`
  - Build a new Bystro database: `docker run bystro-annotator bystro-build.pl --help`
  - Fetch dependencies, before building: `docker run bystro-annotator bystro-utils.pl --help`

### Installing Bystro Annotator on Bare Metal / Directly on Host Operating System

The easiest way to install Bystro directly on your machine is to run:

- Debian/Ubuntu: `../install-apt.sh`
- Centos/Fedora/Amazon Linux: `../install-rpm.sh`

You will be prompted for "sudo" access to install the necessary system level dependencies.

### Manual/Custom Install

The previous instructions configured a local copy of Perl for you, using Perlbrew. If you want to use your system's Perl,
or otherwise control the installation process, follow the "Manual/Custom Install" instructions to give you greater control over installation.

Else, just skip to the next section [Configure Bystro Annotator](#configuring-the-bystro-annotator).

First you'll need to install some prerequisites:

- Debian/Ubuntu: `sudo ../install/install-apt-deps.sh`
- Centos/Fedora/Amazon Linux: `sudo ../install/install-rpm-deps.sh`
- bgzip: `../install/install-htslib.sh ~/.profile ~/.local`

Bystro relies on a few `Go` programs, which can be installed with the following:

```bash
# Where to install the Bystro Go programs (will go into ~/.local/go in this case)
BYSTRO_GO_PROGRAMS_INSTALL_DIR=~/.local
# Where to install Go itself (will go into ~/go in this case)
GOLANG_BINARY_INSTALL_DIR=~/
# Where to add the Go binaries to your PATH
PROFILE_PATH=~/.profile
# Where Bystro is installed
BYSTRO_INSTALL_DIR=~/bystro
# The platform to install Go for
GO_PLATFORM=linux-amd64
# The version of Go to install
GO_VERSION=1.21.4

# BYSTRO_GO_PROGRAMS_INSTALL_DIR and GO_BINARY_INSTALL_DIR directories must exist
mkdir -p $BYSTRO_GO_PROGRAMS_INSTALL_DIR
mkdir -p $GOLANG_BINARY_INSTALL_DIR

# Assuming we are installing this on linux, on an x86 processor
# and that our login shell environment is stored in ~/.profile (another common one is ~/.bash_profile)
../install/install-go.sh $PROFILE_PATH $GOLANG_BINARY_INSTALL_DIR $BYSTRO_GO_PROGRAMS_INSTALL_DIR $BYSTRO_INSTALL_DIR $GO_PLATFORM $GO_VERSION

source ~/.profile
```

Next, we need to install the Bystro Perl library and its Perl dependencies. The instructions for installing the Bystro Perl library use [`cpm`](https://metacpan.org/pod/App::cpanminus).

- Alternatively you can use [cpanm](https://metacpan.org/dist/App-cpanminus/view/bin/cpanm), which can be installed with the following: `curl -fsSL https://cpanmin.us | perl - App::cpanminus`
- Just replace every `cpm install --test` and `cpm install` command with `cpanm`

<br>

To install `cpm`, run the following:

```bash
# Install cpm
curl -fsSL https://raw.githubusercontent.com/skaji/cpm/main/cpm | perl - install App::cpm
```

You will need to configure where Perl stores its libraries. By default, `cpm` will install libraries in `./local` in the current directory.

- You will need to make sure that this path is in your `PERL5LIB` environment variable:

  ```bash
  # Assuming you were in the ~/bystro/perl directory when you ran `cpm install`, you would get a folder `~/bystro/perl/local` with the libraries and binaries cpm installed
  # We need to add this to our PERL5LIB and PATH environment variables
  # You would put these commands in your ~/.profile or ~/.bash_profile
  export PERL5LIB=~/bystro/perl/local/lib/perl5:$PERL5LIB
  export PATH=~/bystro/perl/local/bin:$PATH
  ```

- If you want to install libraries and binaries into a different local directory, replace `cpm install` with `cpm install -L=/path/to`, which will cause libraries to be installed in `/path/to/lib/perl5` and binaries into `/path/to/bin`. You will need to make sure that these paths are in your `PERL5LIB` and `PATH` environment variables respectively:

  ```bash
  # Assuming you ran `cpm install -L=/path/to` for all of your cpm install commands
  # Put this in your ~/.profile or ~/.bash_profile
  export PERL5LIB=/path/to/lib/perl5:$PERL5LIB
  export PATH=/path/to/bin:$PATH
  ```

- If you want to install libraries in the default Perl library path, as specified by Perl's @INC, replace the `cpm install` commands with `cpm install -g`

<br>

A few dependencies must be specially separately installed:

```bash
cpm install --test https://github.com/bystrogenomics/msgpack-perl.git

ALIEN_INSTALL_TYPE=share cpm install --test Alien::LMDB
cpm install --test LMDB_File

# no --test option because it has a trivial failure related to formatting of cli help strings
cpm install MouseX::Getopt
```

However, if you are using Perl > 5.36.0, you will need to manually install LMDB_File 0.14, which will require `make`:

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

<br>

Now you're ready to try Bystro:

```bash
# First let's run our test suite
cd ~/bystro/perl
prove -r ./t -j$(nproc)

# Then let's try running bystro-annotate.pl
bystro-annotate.pl --help

# Expected output
# usage: bystro-annotate.pl [-?cio] [long options...]
#     --[no-]help (or -?)        Prints this usage information.
#                                aka --usage
#     --input STR... (or -i)     Input files. Supports mulitiple files:
#                                --in file1 --in file2 --in file3
#                                aka --in
#     --output STR (or -o)       Base path for output files: /path/to/output
#                                aka --out
#     --[no-]json                Do you want to output JSON instead?
#                                Incompatible with run_statistics
#     --config STR (or -c)       Yaml config file path.
#                                aka --configuration
#     --overwrite INT            Overwrite existing output file.
#     --[no-]read_ahead          For dense datasets, use system read-ahead
#     --debug NUM
#     --verbose INT
#     --compress STR             Enable compression. Specify the type of
#                                compression: lz4 gz bgz. `bgz` is an alias
#                                for gz (gzip); when bgzip is available, it
#                                will be used and will generate a block
#                                gzipped file with index
#     --[no-]archive             Place all outputs into a tarball?
#     --run_statistics INT       Create per-sample feature statistics (like
#                                transition:transversions)?
#     --delete_temp INT          Delete the temporary directory made during
#                                annotation
#     --wantedChr STR            Annotate a single chromosome
#                                aka --chr, --wanted_chr
#     --maxThreads INT           Number of CPU threads to use (optional)
#                                aka --threads
#     --publisher STR            Tell Bystro how to send messages to a
#                                plugged-in interface (such as a web
#                                interface)
#     --[no-]ignore_unknown_chr  Don't quit if we find a non-reference
#                                chromosome (like ChrUn)
#     --json_config STR          JSON config file path. Use this if you
#                                wish to invoke the annotator by file
#                                passing.
#     --result_summary_path STR  Where to output the result summary.
#                                Defaults to STDOUT
```

## Configuring the Bystro Annotator

Once Bystro is installed, we need to download a database for the species/assembly we're going to be analyzing and then configure the Bystro Annotator to use it.

Database configurations are stored in YAML files in the `config` directory. By default Bystro ships with configurations for human genome assemblies hg19 (~/bystro/config/hg19.yml) and hg38 (~/bystro/config/hg38.yml), though you can create your own configurations for other species/assemblies.

### Example Configuration

1. Download and unpack the human hg38 Bystro database

   ```bash
   MY_DATABASE_DIR=/mnt/annotator
   sudo mkdir -p $MY_DATABASE_DIR
   sudo chown -R $USER:$USER $MY_DATABASE_DIR
   cd $MY_DATABASE_DIR
   wget https://s3.amazonaws.com/bystro-db/hg38_v11.tar.gz
   bgzip -d -c --threads 32 hg38_v11.tar.gz | tar xvf -
   ```

   - You can chooose a directory other than `/mnt/annotator/`; that is just the default expected by ~/bystro/config/hg38.yml. If you choose something else, just update the `database_dir` property in the configuration file

     - with `yq`:

       ```bash
       MY_DATABASE_DIR=/path/somewhere/else
       # Assuming we downloaded and unpacked the database to /path/somewhere/else/hg38_v11
       # Update the database_dir property in the configuration file using `yq`
       # You can also do this manually by editing the configuration file (in this example ~/bystro/config/hg38.yml)
       yq write -i ~/bystro/config/hg38.yml database_dir $MY_DATABASE_DIR/hg38_v11
       ```

   - `tar` is required to unpack the database, which is stored as a compresssed tarball, but you can unzip the tarball uzing `gzip -d -c` instead of `bgzip -d -c --threads 32` if you don't have `bgzip` installed. It will work, just slower.

   - You need ~691GB of free space for hg38 and ~376GB of free space for hg19, including the space for the tar.gz archives.

   - The unpacked databases are ~517GB for hg38 and ~283GB for hg19.

2. (optional) Configure your Bystro Annotator to use a temporary directory with fast local storage, by editing the configuration files `tmp_dir` property to a directory on your fast local storage. This directory must be writable by the user running bystro-annotate.pl.

   If you've installed `yq` this is easy:

   ```bash
   MY_FAST_LOCAL_TEMP_STORAGE_FOLDER=/mnt/annotator/tmp
   mkdir -p $MY_FAST_LOCAL_STORAGE

   # Or edit ~/bystro/config/hg38.yml file manually
   yq write -i ~/bystro/config/hg38.yml temp_dir $MY_FAST_LOCAL_TEMP_STORAGE_FOLDER
   ```

   If temp_dir is not set, the files will be written directly to the output directory (see `--output` option in `bystro-annotate.pl`).

## Databases

1. [Human (hg38) database](https://s3.amazonaws.com/bystro-db/hg38_v11.tar.gz)
2. [Human (hg19) database](https://s3.amazonaws.com/bystro-db/hg19_v10.tar.gz)
3. [Rat (rn7) database](https://s3.amazonaws.com/bystro-db/rn7.tar.gz)
4. There are no restrictions on species support, but for the open source Bystro Annotator we currently only build human and rat genomes, and do not guarantee that the open-source version will be up to date. Please create a GitHub issue if you would like us to support others or need updates to the current databases.

## Running your first annotation

Example: Annotate an hg38 VCF file:

```sh
bystro-annotate.pl --config ~/bystro/config/hg38.yml --threads 32 --input gnomad.genomes.v4.0.sites.chr22.vcf.bgz --output test/my_annotation --compress gz
```

The above command will annotate the `gnomad.genomes.v4.0.sites.chr22.vcf.bgz` file with the hg38 database, using 32 threads, and output the results to `test`, and will use `my_annotation` as the prefix for output files.

The result of this command will be:

```sh
Created completion file
{
   "error" : null,
   "totalProgress" : 8599234,
   "totalSkipped" : 0,
   "results" : {
      "header" : "my_annotation.annotation.header.json",
      "sampleList" : "my_annotation.sample_list",
      "annotation" : "my_annotation.annotation.tsv.gz",
      "dosageMatrixOutPath" : "my_annotation.dosage.feather",
      "config" : "hg38.yml",
      "log" : "my_annotation.annotation.log.txt",
      "statistics" : {
         "qc" : "my_annotation.statistics.qc.tsv",
         "json" : "my_annotation.statistics.json",
         "tab" : "my_annotation.statistics.tsv"
      }
   }
}
```

Explanation of the output:

- `my_annotation.annotation.header.json`: The header of the annotated dataset
- `my_annotation.sample_list`: The list of samples in the annotated dataset
- `my_annotation.annotation.tsv.gz`: A block gzipped TSV file with one row per variant and one column per annotation. Can be decompressed with `bgzip` or any program compatible with the gzip format, like `gzip` and `pigz`.
- `my_annotation.dosage.feather`: The dosage matrix file, where the first column is the `locus` column in the format "chr:pos:ref:alt", and columns following that are sample columns, with the dosage of the variant for that sample (0 for homozygous reference, 1 for 1 copy of the alternate allele, 2 for 2, and so on). -1 indicates missing genotypes. The dosage is the expected number of alternate alleles, given the genotype. This is useful for downstream analyses like imputation, or for calculating polygenic risk scores
  - This file is in the [Arrow feather format](https://arrow.apache.org/docs/python/feather.html), also known as the "IPC" format. This is an ultra-efficient format for machine learning, and is widely supported, in Python libraries like [Pandas](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_feather.html), [Polars](https://docs.pola.rs/api/python/stable/reference/api/polars.read_ipc.html), [PyArrow](https://arrow.apache.org/docs/python/generated/pyarrow.feather.read_feather.html), as well as languages like [R](https://arrow.apache.org/docs/r/reference/read_feather.html) and [Julia](https://github.com/apache/arrow-julia)
- `hg38.yml`: The configuration file used for the annotation. You can use this to either re-build the Bystro database from scratch, or to re-run the annotation with the same configuration
- `my_annotation.annotation.log.txt`: The log file for the annotation
- `my_annotation.statistics.tsv`: A TSV file with sample-wise statistics on the annotation
- `my_annotation.statistics.qc.tsv`: A TSV file that lists any samples that failed quality control checks, currently defined as being outside 3 standard deviations from the mean on any of the sample-wise statistics
- `my_annotation.statistics.json`: A JSON file with the same sample-wise statistics on the annotation
- 'totalProgress': The number of variants processed; this is the number of variants passed to the Bystro annotator by the bystro-vcf pre-processor, which performs primary quality control checks, such as excluding sites that have no samples with non-missing genotypes, or which are not FILTER=PASS in the input VCF. We also exclude sites that are not in the Bystro database, and sites that are not in the Bystro database that are not in the input VCF. In more detail:
  - Variants must have FILTER value of PASS or " . "
  - Variants and ref must be ACTG (no structural variants retained)
  - Multiallelics are split into separate records, and annotated separately
  - MNPs are split into separate SNPs and annotated separately
  - Indels are left-aligned
  - The first base of an indel must be the reference base after multiallelic decomposition and left-alignment
  - If genotypes are provided, entirely missing sites are dropped

## Developer Resources

### Coding style and tidying

The `.perltidyrc` gives the coding style and `tidyall` from [Code::TidyAll](https://metacpan.org/dist/Code-TidyAll) can be used to tidy all files with `tidyall -a`.
Please tidy all files before submitting patches.

Install tidyall, perltidy, and perlcritic like so:

```bash
cpanm Code::TidyAll Perl::Tidy Perl::Critic
```
