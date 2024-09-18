# Bystro Perl package


## Installing Bystro using docker

To build a docker image using the `Dockerfile`, run the following:

```bash
docker build --tag bystro-cli .
```

## Installing Bystro using `cpam`

The instructions for installing Bystro locally uses [`cpanm`](https://metacpan.org/pod/App::cpanminus).

Assuming that you've cloned the repository and are working on it locally, then the dependencies can mostly be installed with cpanm.
But there are a few one-off dependencies that require a slightly modified approach.

One-off dependencies can be installed as follows:

```bash
cpanm --quiet https://github.com/bystrogenomics/msgpack-perl.git
cpanm --quiet --notest MouseX::Getopt
git clone --depth 1 --recurse-submodules https://github.com/salortiz/LMDB_File.git \
  && cd LMDB_File \
  && cpanm --quiet . \
  && cd .. \
  && rm -rf LMDB_File
# NOTE: you will need mysql_config to install this
#       ubuntu 22.04 LTS => sudo apt install -y libmariadb-dev libmariadb-dev-compat
#       amazon 2023 => sudo yum install -y <mariadb105>
cpanm --quiet DBD::MariaDB@1.23
```

The remaining dependencies are installed like this:

```bash
cpanm --installdeps .
```

After installing dependencies, use `prove -lr t` to run tests.

## Installing Bystro locally using dzil

Bystro uses [`Dist::Zilla`](https://github.com/rjbs/dist-zilla) for packaging and is configured with `dist.ini`.
This approach requires installing `Dist::Zilla` and author dependencies and one off-dependencies described in the  above.

```bash
# Install Dist::Zilla and Archive::Tar::Wrapper (to slightly speed up building)
cpanm --quiet Dist::Zilla Archive::Tar::Wrapper

# Install build dependencies
dzil authordeps --missing | cpanm --quiet

# Install Bystro dependencies
dzil listdeps --missing | cpanm --quiet

# Install Bystro
dzil install
```

## Install Bystro using `cpm`

Install [cpm](https://metacpan.org/pod/App::cpm) with `curl -fsSL https://raw.githubusercontent.com/skaji/cpm/main/cpm | perl - install -g App::cpm`.

NOTE: you will need mysql_config to install this:
   ubuntu 22.04 LTS => sudo apt install -y libmariadb-dev libmariadb-dev-compat
   amazon 2023 => sudo yum install -y <mariadb105>

```bash
# install msgpack fork
cpm install -g https://github.com/bystrogenomics/msgpack-perl.git

# install MouseX::Getopt despite some tests failing
cpm install -g --no-test MouseX::Getopt

# install LMDB_File that comes with latest LMDB
# NOTE: you will need mysql_config to install this
#       ubuntu 22.04 LTS => sudo apt install -y libmariadb-dev libmariadb-dev-compat
#       amazon 2023 => sudo yum install -y <mariadb105>
git clone --depth 1 --recurse-submodules https://github.com/salortiz/LMDB_File.git \
  && cd LMDB_File \
  && cpanm . \
  && cd .. \
  && rm -rf LMDB_File

# clone bystro and change into perl package
git clone git@github.com:bystrogenomics/bystro.git && cd bystro/perl

# install dependencies
cpm install -g --with-develop

# Install dzil build dependencies
cpm install -g --show-build-log-on-failure $(dzil authordeps --missing)
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