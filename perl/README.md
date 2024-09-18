# Bystro Annotator Package

## Installing Bystro Annotator using Docker

To build a Docker image using the `Dockerfile`, run the following:

```bash
cd ../ && docker build -t bystro-annotator -f Dockerfile.perl .
```

## Installing Bystro Annotator Bare Metal / Locally

First you'll need to install some prerequisites:

- Debian/Ubuntu: 'sudo ../install/install-rpm-deps.sh`
- Centos/Fedora/Amazon Linux: 'sudo ../install/install-apt-depts.sh`

The instructions for installing Bystro locally use [`cpm`](https://metacpan.org/pod/App::cpanminus).

- Alternatively you can use [cpanm](https://metacpan.org/dist/App-cpanminus/view/bin/cpanm)
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

Another alternative package manager is [`cpanm`](https://metacpan.org/pod/App::cpanminus), which can be installed with the following:

```bash
curl -fsSL https://cpanmin.us | perl - App::cpanminus
```

- Please read the [cpanm documentation](https://metacpan.org/pod/App::cpanminus) for more information on how to use it.
- Briefly, you would replace every `cpm install --test` and `cpm install` command with `cpanm`

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
    && cpm install --show-build-log-on-failure . \
    && cd ..
  ```

Now you can install the rest of the dependencies:

```bash
  cpm install
```

Bystro also relies on a few `Go` programs, which can be installed with the following:

```bash
mkdir ~/.local

# Assuming we are installing this on linux, on an x86 processor
# and that our login shell environment is stored in ~/.profile (another common one is ~/.bash_profile)
../install/install-go.sh ~/.profile ~/ ~/.local ~/bystro/ linux-amd64 1.21.4

source ~/.profile
```

Now you're ready to try Bystro:

```bash
# First let's run our test suite
cd ~/bystro/perl
prove -r ./t -j$(nproc)

# Then let's try running bystro-annotate.pl
bystro-annotate.pl --help
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
