# Bystro Annotator

## Building Bystro perl package

Bystro uses Dist::Zilla for packaging.

### Local Installation

#### Install required external dependencies

Bystro runs on a unix-like environment, e.g. linux or macOS.
It requires mysql or mariadb, bzip2, lz4, tar, LMDB, and openssl.

```bash
# on aws linux 2023 the required packages are installed by default except mariadb (alternative to mysql)
sudo yum install -y mariadb105-devel.x86_64
```

#### (Optional) Install Perl via perlbrew

It is optional but recommended to install Perl using perlbrew rather than using the version of Perl that comes with the OS.

#### Install Bystro Perl build dependencies

Building or installing bystro from this repository requires `Dist::Zilla` and author dependencies.

```bash
# system install
cpanm Dist::Zilla Archive::Tar::Wrapper

# install author dependencies using dzil
dzil authordeps | cpanm
```

#### Install one-off Bystro Perl dependencies

Bystro requires a few perl packages that are currently installed outside of the normal build process that are installed as shown below.

```bash
# install one-off dependencies
cpanm https://github.com/bystrogenomics/msgpack-perl.git
cpanm --force MouseX::Getopt
git clone --recurse-submodules https://github.com/salortiz/LMDB_File.git \
    && cd LMDB_File \
    && perl Makefile.PL \
    && make test \
    && make install
```

#### Install Bystro

With the build and one-off dependencies installed, install Bystro using `dzil`.

```bash
# build and install package on system
dzil install
```

### Docker Installation

Building a docker container with bystro requires local installation of the build dependencies (but not the external dependencies).

```bash
# build package and install on container
dzil build && docker build -t bystro .
```

## Code tidying and analysis

Bystro uses `Perl::Tidy` and `Perl::Critic` for linting and analysis. `Code::Tidyall` recursively tidies all code using `Perl::Tidy`.

```bash
# install linters and code analyzer
cpanm Perl::Critic Code::TidyAll Perl::Tidy

# tidy code
tidyall -a
```
