# Bystro Annotator Package

## Installing Bystro Annotator using Docker

To build a Docker image using the `Dockerfile`, run the following:

```bash
cd ../ && docker build -t bystro-annotator -f Dockerfile.perl .
```

## Installing Bystro Annotator Bare Metal / Locally

The instructions for installing Bystro locally use [`cpm`](https://metacpan.org/pod/App::cpanminus).

To install `cpm`, run the following:

- If using the system Perl:

  ```bash
  curl -fsSL https://raw.githubusercontent.com/skaji/cpm/main/cpm | sudo perl - install -g App::cpm
  ```

- If using local Perl through [perlbrew](https://perlbrew.pl) or [plenv](https://github.com/tokuhirom/plenv), or system Perl but with [local::lib](https://srcc.stanford.edu/farmshare/software-perllocallib) to ensure that libraries are installed locally (thus not requiring `sudo privileges`):

  ```bash
  curl -fsSL https://raw.githubusercontent.com/skaji/cpm/main/cpm | perl - install -g App::cpm
  ```

Assuming that you've cloned the repository and are working on it locally, then the dependencies can mostly be installed with cpm.
But there are a few one-off dependencies that require a slightly modified approach.

One-off dependencies can be installed as follows:

```bash
cpm install -g --test https://github.com/bystrogenomics/msgpack-perl.git

ALIEN_INSTALL_TYPE=share cpm install -g --test Alien::LMDB
cpm install -g --test LMDB_File
cpm install -g MouseX::Getopt # fails due to differences from expected output; unimportant
```

- Please note, that if you are using Perl > 5.36.0, you will need to manually install LMDB_File 0.14, which will require `make`

  ```bash
  ALIEN_INSTALL_TYPE=share cpm install -g --test Alien::LMDB
  git clone --depth 1 --recurse-submodules https://github.com/salortiz/LMDB_File.git \
    && cd LMDB_File \
    && git checkout 34acb71d7d86575fe7abb3f7ad95e8653019b282 \
    && perl Makefile.PL && make distmeta \
    && ln -s MYMETA.json META.json && ln -s MYMETA.yml META.yml \
    && cpm install --show-build-log-on-failure -g . \
    && cd ..
  ```

- Alternatively you can use [cpanm](https://metacpan.org/dist/App-cpanminus/view/bin/cpanm) to install the LMDB_File 0.14 dependencies:

  ```bash
    ALIEN_INSTALL_TYPE=share cpm install -g --test Alien::LMDB
    git clone --depth 1 --recurse-submodules https://github.com/salortiz/LMDB_File.git \
      && cd LMDB_File \
      && cpanm . \
      && cd .. \
      && rm -rf LMDB_File
  ```

One of the dependencies, DBD::MariaDB, requires mysql_config to be installed. From DBD::MariaDB documentation:

```
MariaDB/MySQL
You need not install the actual MariaDB or MySQL database server, the client files and the development files are sufficient. They are distributed either in Connector/C package or as part of server package. You need at least MySQL version 4.1.8.

For example, Fedora, RedHat, CentOS Linux distribution comes with RPM files (using YUM) mariadb-devel, mariadb-embedded-devel, mysql-devel or mysql-embedded-devel (use yum search to find exact package names). Debian and Ubuntu comes with DEB packages libmariadb-dev, libmariadbclient-dev, libmariadbd-dev, libmysqlclient-dev or libmysqld-dev (use apt-cache search to find exact package names).

In some cases MariaDB or MySQL libraries depends on external libpcre, libaio, libnuma, libjemalloc or libwrap libraries. If it is that case, they needs to be installed before MariaDB/MySQL libraries.

These are sufficient, if the MariaDB/MySQL server is located on a foreign machine. You may also create client files by compiling from the MariaDB/MySQL source distribution and using
```

We provide the commands to install the dependencies for Ubuntu 22.04 LTS and Amazon Linux 2023, see `../install/install-apt-deps.sh` for Debian based systems like Ubuntu and `../install/install-rpm-deps.sh` for Centos/RedHat based systems like Amazon Linux 2023.

Once the dependencies are installed, you can install the remaining dependencies with `cpm` and `dzil`:

## Installing Bystro locally using dzil

Bystro uses [`Dist::Zilla`](https://github.com/rjbs/dist-zilla) for packaging and is configured with `dist.ini`.
This approach requires installing `Dist::Zilla` and author dependencies and one off-dependencies described in the above.

```bash
# Install cpm
curl -fsSL https://raw.githubusercontent.com/skaji/cpm/main/cpm | perl - install -g App::cpm

# Install Dist::Zilla and Archive::Tar::Wrapper (to slightly speed up building)
cpm install -g Dist::Zilla Archive::Tar::Wrapper

# Install all dependencies
dzil authordeps --missing | cpm install -g
dzil listdeps --author --missing | cpm install -g

# To install Bystro annotator packages into your PATH
dzil install

# Or alternatively, make sure that the `bin` directory is in your PATH
export PATH=~/bystro/perl/bin:$PATH
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
