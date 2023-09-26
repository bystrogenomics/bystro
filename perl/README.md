# Bystro Annotator

## Build a release

Releases are generated using Dist::Zilla.

### Install author development dependencies

```bash
# on aws linux 2023
sudo yum install -y mariadb105-devel.x86_64

# system install
cpanm Dist::Zilla Archive::Tar::Wrapper

# install author dependencies
dzil authordeps | cpanm

# install one-off dependencies
cpanm https://github.com/bystrogenomics/msgpack-perl.git
cpanm --force MouseX::Getopt
git clone --recurse-submodules https://github.com/salortiz/LMDB_File.git \
    && cd LMDB_File \
    && perl Makefile.PL \
    && make test \
    && make install

# build and install package on system
dzil install

# build package and install on container
dzil build && docker build -t bystro .
```

## Code Tidying

```bash
cpanm Code::TidyAll Perl::Critic Perl::Tidy
```
