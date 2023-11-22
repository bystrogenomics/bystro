# Bystro Perl package


## Installing Bystro using docker

To build a docker image using the `Dockerfile`, run the following:

```bash
docker build --tag bystro-cli .
```

## Installing Bystro using `cpanm`

The instructions for installing Bystro locally uses [`cpanm`](https://metacpan.org/pod/App::cpanminus).

Assuming that you've cloned the repository and are working on it locally, then the dependencies can mostly be installed with cpanm.
But there are a few one-off dependencies that require a slightly modified approach.

One-off dependencies can be installed as follows:

```bash
# install msgpack fork
cpanm --quiet https://github.com/bystrogenomics/msgpack-perl.git

# install MouseX::Getopt despite some tests failing
cpanm --quiet --notest MouseX::Getopt

# install LMDB_File that comes with latest LMDB
git clone --depth 1 --recurse-submodules https://github.com/salortiz/LMDB_File.git \
  && cd LMDB_File \
  && cpanm --quiet . \
  && cd .. \
  && rm -rf LMDB_File

# install mysql driver
# NOTE: you will need mysql_config to install this
#       ubuntu 22.04 LTS => sudo apt install -y libmariadb-dev libmariadb-dev-compat
#       amazon 2023 => sudo yum install -y mariadb105-devel.x86_64
cpanm --quiet DBD::mysql@4.051
```

The remaining dependencies are installed like this:

```bash
cpanm --installdeps .
```

## Install Bystro using `cpm`

Install [cpm](https://metacpan.org/pod/App::cpm) with `curl -fsSL https://raw.githubusercontent.com/skaji/cpm/main/cpm | perl - install -g App::cpm`.

```bash
# install msgpack fork
cpm install -g https://github.com/bystrogenomics/msgpack-perl.git

# install MouseX::Getopt despite some tests failing
cpm install -g --no-test MouseX::Getopt

# install LMDB_File that comes with latest LMDB
git clone --depth 1 --recurse-submodules https://github.com/salortiz/LMDB_File.git \
  && cd LMDB_File \
  && cpanm . \
  && cd .. \
  && rm -rf LMDB_File

# install mysql driver
# NOTE: you will need mysql_config to install this
#       ubuntu 22.04 LTS => sudo apt install -y libmariadb-dev libmariadb-dev-compat
#       amazon 2023 => sudo yum install -y mariadb105-devel.x86_64
cpm install -g DBD::mysql@4.051

# clone bystro and change into perl package
git clone git@github.com:bystrogenomics/bystro.git && cd bystro/perl

# install dependencies (uses cpanfile)
cpm install -g --with-develop
```

After installing dependencies, use `prove -lr t` to run tests.

## Installing development tools

These are tools needed for packaging and releasing Bystro's Perl package, which are not needed for normal testing or development.

Bystro uses [`Dist::Zilla`](https://github.com/rjbs/dist-zilla) for packaging and is configured with `dist.ini`.
This approach requires installing `Dist::Zilla` and author dependencies and one off-dependencies described in the  above.

```bash
# Install Dist::Zilla and Archive::Tar::Wrapper (to slightly speed up building)
cpanm --quiet Dist::Zilla Archive::Tar::Wrapper
# - or -
cpm install -g Dist::Zilla Archive::Tar::Wrapper

# Install build dependencies
dzil authordeps --missing | cpanm --quiet
# - or - 
cpm install -g $(dzil authordeps --missing)

# Install Bystro Package
dzil install

# Test Bystro Package
dzil test --all
```

## Coding style and tidying

The `.perltidyrc` gives the coding style and `tidyall` from [Code::TidyAll](https://metacpan.org/dist/Code-TidyAll) can be used to tidy all files with `tidyall -a`.
Please tidy all files before submitting patches.

Install tidyall, perltidy, and perlcritic like so

```bash
cpanm Code::TidyAll Perl::Tidy Perl::Critic
```

## Specifying libraries in Perl

To specify specific libraries for the Perl codebase, use `use <lib> <version>` (see [use documentation](https://perldoc.perl.org/functions/use)).
Packaging with `Dist::Zilla` will specify them in the `Makefile.PL` that it creates.
