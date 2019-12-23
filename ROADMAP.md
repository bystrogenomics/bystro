# Bystro 2020 Roadmap (wip)

## Goals
1. Upgrade databases
2. Migrate users to master branch (cut Bystro 1.0 release)
  * Before release:
    * Annotate directly from URL (streaming using POSIX pipe, for s3)
    * Improve error messages back to the user
    * Check test coverage
3. Support custom annotation sources (integrate akotlar/genpro updates)
4. Distributed pipeline submissions (run arbitrary code on Bystro annotations)backed by beanstalkd
5. Web improvements:
  * UI: Autocomplete search, tooltips, better listing of available annotation fields, visualizations
  * Allow annotation by chromosome: position, and make the page indexable by Google
6. Integration of important statistical tests (single-variant association, burden tests, variance-component tests).
7. Infrastructure improvements:
  * Upgrade beanstalkd version
  * Evaluate AWS Fargate
  * Auto-scaling stateful servers, spot-market transient workers. Try to avoid Kubernetes
  * CI (auto-test) for easier PR evaluation.

## Secondary targets (time permitting, or when blocked on Goals)
1. Migrate to Rust version of bystro-vcf (vcf2 branch)
2. Native Rust blocked-gzip decompression
  * To be used in bystro-vcf
  * Release as Rust package and on pypi using PyO3
3. Format changes to column-oriented store, Rust client (with Python wrapping) to efficiently select data 
4. Complete migrate from Angular to React (NextJS, akotlar/bystro:web2 branch)

## Long-term goals
1. Migrate remainder of Bystro to Rust or Go. Client should be prioritized.
  * less important because the Perl client is typically not a CPU bottleneck (processing the VCF file, annotating genotypes is)
  * more efficient IPC to Perl process (larger pipe buffer for instance) is however important, and should be evaluated in advance of anny move from Perl to Rust.
  

