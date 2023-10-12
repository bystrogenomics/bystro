# Bystro Perl package

## Installing Bystro using docker

To build a docker image using the `Dockerfile`, run the following:

```bash
docker build --tag bystro-cli .
```

## Installing Bystro dependencies locally

Assuming that you've cloned the repository and are working on it locally, then the dependencies can mostly be installed with cpanm.
But there are a few one-off dependencies that require a slightly modified approach.

One-off dependencies can be installed as follows:

```bash
cpanm --quiet https://github.com/bystrogenomics/msgpack-perl.git
cpanm --quiet --notest MouseX::Getopt
git clone --depth 1 -branch v0.12 --recurse-submodules https://github.com/salortiz/LMDB_File.git \
  && cd LMDB_File \
  && cpanm --quiet .
  && cd .. \
  && rm -rf LMDB_File
cpanm --quiet DBD::mysql@4.051
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

## Coding style and tidying

The `.perltidyrc` gives the coding style and `tidyall` from [Code::TidyAll](https://metacpan.org/dist/Code-TidyAll) can be used to tidy all files with `tidyall -a`.
Please tidy all files before submitting patches.

## Specifying libraries in Perl

To specify specific libraries for the Perl codebase, use `use <lib> <version>` (see [use documentation](https://perldoc.perl.org/functions/use)).
Packaging with `Dist::Zilla` will specify them in the `Makefile.PL` that it creates.
