# Bystro Annotator

## Install

```bash
# system install
cpanm Dist::Zilla
dzil install

# container
dzil build && docker build -t bystro .
```

## Author dependencies

```bash
# Dist::Zilla is used to manage the packaging
cpanm Dist::Zilla
dzil authordeps | cpanm
```

## Code Tidying

```bash
cpanm Code::TidyAll Perl::Critic Perl::Tidy
```
