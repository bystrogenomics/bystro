# Bystro Open Source

At Bystro, we believe natural language is the right interface for genetic and proteomic analysis. We are building the world's first LLM-powered natural language analysis engine that takes your questions about complex genetic and proteomic datasets, and converts them into statistical answers with easy to understand summaries and visualizations.

This is our open-source repo of machine learning methods for high dimensional statistics, as well as some applications in genomics and proteomics.

This work is the basis for the Bystro natural language analysis platform for genetics & proteomics. See **https://bystro.io**

<br/>

## Machine Learning Methods

We are working hard on cutting edge algorithms, and haven't found much time for documentation. More detailed descriptions coming soon, but until then, a brief summary is found below:

### Covariance Matrix Estimation and Hypothesis Testing

```python
from bystro.covariance import *
```

1. Regularized covariance matrix estimation methods well suited for smaller sample size regimes where n << p
2. Covariance matrix hypothesis tests, like the 2 sample covariance test (`from bystro.random_matrix_theory.rmt4ds_cov_test import two_sample_cov_test`)

### Random Matrix Theory Methods

```python
from bystro.random_matrix_theory import *
```

Random Matrix Theory modules that are foundational for significance tests, such as our `two_sample_cov_test`

### Stochastic Gradient Langevin

```python
from bystro.stochastic_gradient_langevin import *
```

Implementation of Stochastic Gradient Langevin algorithm in https://www.ics.uci.edu/~welling/publications/papers/stoclangevin

### Fair Machine Learning and Supervised PPCA / Variational Principal Component Regression

```python
from bystro.supervised_ppca import *
```

`supervised_ppca` is a collection of generative methods:

1. Probabilistic PCA (PPCA)
2. Supervised PPCA (also know as Variational Principal Component Regression): Novel method for network analysis that is able to pick up dynamics of interest in low variance components. Also competitive with Elastic Net in a regression context, without shrinking covariates (instead shrinks them in latent space). See our recent publication: https://arxiv.org/abs/2409.02327
3. Adversarial Probabilistic PCA: Fair ML method that removes the influence of M sensitive variables (confounding factors), from high dimensional data

<br>

## Applications in Proteomics

For proteomics analyses, we have 4 modular steps that can be applied alone or combined together:

1. Imputation   
    - Recommended if there are missing values within your data, demo here: [Soft Impute Demo](python/python/bystro/examples/Demo_Imputing_Missing_Data.ipynb)
2. Batch Correction
    - Recommended for TMT data that has a control per batch to correct for small batches, demo here: [Small sample batch correction Demo](python/python/bystro/examples/Demo_accounting_for_batch_effects_with_control.ipynb)
3. Removal of Confounding factors
    - Recommended if there are confounding factors such as sex or ancestry that you want to remove from your data, demo here: [Fair PCA Demo](python/python/bystro/examples/Demo_Fair_PCA.ipynb)
4. Network Analysis
    - Recommended if you want to discover predictive networks in your proteomic data, demo here: [Learning Predictive Network Demo](python/python/bystro/examples/Demo_Learning_Predictive_Network.ipynb)
<br>

## Applications in Genetics

Make genetic results more generalizable by removing information from confounding factors:
  - Remove ancestry-related information in multi-ancestry cohorts to reduce bias
  - Remove effect of batch in meta-analyses
  - See demo here: [Fair PCA Demo](python/python/bystro/examples/Demo_Fair_PCA.ipynb)
<br>

## Combined (Multi-omics) Applications

Combine genomic and proteomic data for any downstream analyses or data exploration
- Read more here: [Proteomics combine README.md](python/python/bystro/proteomics/README.md)

<br>

## Publications

[Talbot et al. arXiv, 2024](https://arxiv.org/abs/2409.02327)

[Kotlar et al, Genome Biology, 2018](https://doi.org/10.1186/s13059-018-1387-3)

<br>

## Installing Bystro Python library

To install the Bystro Python package, run:

```sh
pip install bystro
```

The Bystro ancestry CLI `score` tool (`bystro-api ancestry score`) parses VCF files to generate dosage matrices. This requires `bystro-vcf`, a Go program which can be installed with:

```sh
# Requires Go: install from https://golang.org/doc/install
go install github.com/bystrogenomics/bystro-vcf@2.2.3
```

Bystro is compatible with Linux and MacOS. Windows support is experimental. If you are installing on MacOS as a native binary (Arm), you will need to install the following additional dependencies:

```sh
brew install cmake
```

Please refer to [INSTALL.md](INSTALL.md) for more details.
