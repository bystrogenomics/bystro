# Bystro High Dimensional Genomic Annotator Documentation

## What is the Bystro Annotator?

In order to make use of genetic data, which is composed of variants (colloquially known as mutations) and the individuals (samples) that have those variants, we need to clean that data and then carefully describe those variants with as much accurate information as possible. This information can then be used for qualitative and quantitative studies of the variants/samples. These descriptions are known as labels, model parameters, features, or "annotations", depending on your field. These activites are collectively called data curation and data labeling, but in genetics we call them qc and annotation.

The Bystro Annotator is the fastest and most comprehensive data curation and labeling library in the world for genetic data. It takes 1 or more VCF ([Variant Call Format](https://samtools.github.io/hts-specs/VCFv4.2.pdf)) or SNP ([PEMapper/PECaller](https://www.pnas.org/doi/full/10.1073/pnas.1618065114)) files as input, and outputs a cleaned and thoroughly labeled (annotated) representation of the data, along with a genotype dosage matrix in the [Arrow Feather V2/IPC format](https://arrow.apache.org/docs/python/feather.html), as well as a set of statistics that describe sample-level characteristics of the data.

Bystro Annotator annotates variants as well as sample genotypes. It is capable of processing millions of samples and billions of mutations on commodity hardware such as a laptop or a workstation. It is roughly **100,000** times faster than [Variant Effect Predictor](https://www.ensembl.org/info/docs/tools/vep/index.html) (VEP), **100** times faster than [Annovar](https://doc-openbio.readthedocs.io/projects/annovar/en/latest/), and **50** times faster than [Illumina Connected Annotations](https://developer.illumina.com/illumina-connected-annotations), all while outputting more annotations than any of these tools. What takes VEP years to do, Bystro can do in minutes to hours, all without requiring multiple servers.

Bystro's performance isn't just about speed, it's also about comprehensiveness. In statistics "dimensionality" refers to how many parameters/covariates/descriptions a model has. Imagine that we are modeling a genetic variant to try to understand how it impacts a disease. The parameters in the model are the annotations we have for that variant. Bystro can output thousands of annotations for each variant, and can do so for millions of variants in seconds. This is why we call it "high dimensional".

For exampole, Bystro can afford to provide complete annotations from gnomad v4.1, for all gnomAD populations and subpopulations, from the exomes, genomes, and joint datasets, genome-wide. This means intersecting terabytes of data over the entire genome with each individual variant, all done in microseconds per variant. **No other tool can do this**.

## Running Your First Annotation

See the [INSTALL.md#configuring-the-bystro-annotator](./INSTALL.md#configuring-the-bystro-annotator) section for instructions on how to configure the Bystro Annotator

```sh
bystro-annotate.pl --config ~/bystro/config/hg38.yml --threads 32 --input gnomad.genomes.v4.0.sites.chr22.vcf.bgz --output test/my_annotation --compress gz
```

The above command will annotate the `gnomad.genomes.v4.0.sites.chr22.vcf.bgz` file with the hg38 database, using 32 threads, and output the results to `test`, and will use `my_annotation` as the prefix for output files.

The result of this command will be:

```sh
Created completion file
{
   "error" : null,
   "totalProgress" : 8599234,
   "totalSkipped" : 0,
   "results" : {
      "header" : "my_annotation.annotation.header.json",
      "sampleList" : "my_annotation.sample_list",
      "annotation" : "my_annotation.annotation.tsv.gz",
      "dosageMatrixOutPath" : "my_annotation.dosage.feather",
      "config" : "hg38.yml",
      "log" : "my_annotation.annotation.log.txt",
      "statistics" : {
         "qc" : "my_annotation.statistics.qc.tsv",
         "json" : "my_annotation.statistics.json",
         "tab" : "my_annotation.statistics.tsv"
      }
   }
}
```

Explanation of the output:

- `my_annotation.annotation.header.json`: The header of the annotated dataset

- `my_annotation.sample_list`: The list of samples in the annotated dataset

- `my_annotation.annotation.tsv.gz`: A block gzipped TSV file with one row per variant and one column per annotation. Can be decompressed with `bgzip` or any program compatible with the gzip format, like `gzip` and `pigz`.

- `my_annotation.dosage.feather`: The dosage matrix file, where the first column is the `locus` column in the format "chr:pos:ref:alt", and columns following that are sample columns, with the dosage of the variant for that sample (0 for homozygous reference, 1 for 1 copy of the alternate allele, 2 for 2, and so on). -1 indicates missing genotypes. The dosage is the expected number of alternate alleles, given the genotype. This is useful for downstream analyses like imputation, or for calculating polygenic risk scores

  - This file is in the [Arrow Feather V2 / IPC format](https://arrow.apache.org/docs/python/feather.html), also known as the "IPC" format. This is an ultra-efficient format for machine learning, and is widely supported, in Python libraries like [Pandas](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_feather.html), [Polars](https://docs.pola.rs/api/python/stable/reference/api/polars.read_ipc.html), [PyArrow](https://arrow.apache.org/docs/python/generated/pyarrow.feather.read_feather.html), as well as languages like [R](https://arrow.apache.org/docs/r/reference/read_feather.html) and [Julia](https://github.com/apache/arrow-julia)

- `hg38.yml`: The configuration file used for the annotation. You can use this to either re-build the Bystro Annotation Database from scratch, or to re-run the annotation with the same configuration

- `my_annotation.annotation.log.txt`: The log file for the annotation

- `my_annotation.statistics.tsv`: A TSV file with sample-wise statistics on the annotation

- `my_annotation.statistics.qc.tsv`: A TSV file that lists any samples that failed quality control checks, currently defined as being outside 3 standard deviations from the mean on any of the sample-wise statistics

- `my_annotation.statistics.json`: A JSON file with the same sample-wise statistics on the annotation

- `totalProgress`: The number of variants processed; this is the number of variants passed to the Bystro annotator by the bystro-vcf pre-processor, which performs primary quality control checks, such as excluding sites that have no samples with non-missing genotypes, or which are not FILTER=PASS in the input VCF. We also exclude sites that are not in the Bystro Annotation Database, and sites that are not in the Bystro Annotation Database that are not in the input VCF. In more detail:

  - Variants must have FILTER value of PASS or " . "
  - Variants and ref must be ACTG (no structural variants retained)
  - Multiallelics are split into separate records, and annotated separately
  - MNPs are split into separate SNPs and annotated separately
  - Indels are left-aligned
  - The first base of an indel must be the reference base after multiallelic decomposition and left-alignment
  - If genotypes are provided, entirely missing sites are dropped

## Let's Take a Closer Look at the Annotation Output

The Bystro annotation outputs is a tab-separated file with one header row, and then N rows of annotated variants, one variant per row. The sample genotypes for each variant, and sample-level statistics for each variant are stored in each row in sparse fashion. The annotations are divided into several categories, each of which is described in detail in the [Bystro Annotation Fields](#bystro-annotation-fields) section.

As mentioned, corresponding to the annotation output is a genotype dosage matrix output, which contains the dense representation of genotypes, 1 byte per genotype. It is stored in the [Arrow Feather V2 format](https://arrow.apache.org/docs/python/feather.html), with data compressed using `zstd` compression. Arrow Feather V2 is a columnar datastore, so despite the genotype dosage matrix being typically large, it can be read in a streaming fashion, and even in chunks of samples. We find that we can process thousands of samples within just a few gigabytes of RAM.

## Bystro Annotation Output In Depth

One of the key advantages of Bystro's design is that it outputs data in such a complete manner that it is possible to re-create the source files used for annotation from the Bystro annotation output. Bystro's output formats are therefore designed to retain and reflect complex nested relationships between variant descriptions. Here are key aspects of how we output the data:

1. **Array-Based Fields**: Each annotation field is an array. For fields with multiple values (e.g., transcripts, gene names), the values are separated by semicolons (`;`). The order of the values is maintained across related fields to preserve relationships between the data points. For example:

   - `refSeq.name` might contain `NM1;NM2`, and `refSeq.name2` (gene symbols for these transcripts) could be `Gene1;Gene1`. This ensures the first transcript, `NM1`, corresponds to `Gene1`, and the second transcript, `NM2`, also corresponds to `Gene1`. These relationships are maintained across all fields within a track.

2. **Nested Arrays**: Some fields may be nested arrays, where the individual entries are further divided using forward slashes (`/`). For example, if a transcript has alternate IDs, you may see `NM1a/NM1b;NM2`, indicating two alternate IDs for the first transcript (NM1a and NM1b) and 1 for the 2nd (NM2). This way we can maintain the relationships between the order of fields.

3. **Insertions and Deletions**: For transcript-level annotations like `refSeq.*` (refSeq transcript annotations), `nearest.*`, and `nearestTss.*` (nearest gene by transcript boundaries and by distance to the transcription start site respectively), insertions and deletions affecting multiple nucleotides are separated by pipes (`|`). This allows reporting a transcript consequence per disrupted base.

4. **Reserved Delimiters**: The reserved delimiters described in points 1-3 (`;`, `/`, and `|`) will be stripped and replaced with a comma if found in source inputs to the Bystro Annotation Database.

## What Information Can Bystro Annotator Output?

Bystro Annotator is a general-purpose data curation and labeling engine for genetic data, and has no restrictions on the types of annotations/feature labels it can output. Theoretically it can even support binary data, such as images.

## Variant Representations

Bystro's variant representation deviates slightly from the standard VCF format in the name of simplicity. In particular, it drops the rule that the alternate allele must be ACTG. Dropping this single restriction allows us to represent deletions as occuring at the actual first deleted base, rather than the base before, as in the VCF format. This has a number of knock on benefits:

- The `inputRef` (reference base) in Bystro's annotation outputs is always exactly 1 base long

- The `pos' (position) in Bystro's annotation outputs is always the first affected base, except in the case of insertions, where it is the base before the insertion, since the insertion by definition is between two reference bases

- It is possible to represent all multiallelic site using a single reference base, a single position, and a list of alleles

The Bystro Genotype Dosage Matrix, is a columnar dataset, generated for every collection of VCFs submitted. Its first column is the `locus`, which is `chr:pos:ref:alt`. Every column after that is labeled by the sample name, and contains a -1 for missing genotypes, 0 for reference, 1 for a single alternate allele, 2 for 2 alternate alleles (homozygous in a diploid organism), and so on. It can be used for many things, such as to calculate polygenic risk scores.

### Comparing the VCF and Bystro Variant Representations

Below we'll demonstrate how the Bystro Annotator handles different kinds of variants, using some examples. We'll do these demonstartions using the Bystro VCF preprocessor, which is a Go program used by the Bystro Annotator to convert a VCF into a partially annotated tab-separated output. The Bystro VCF preprocessor is installed with Bystro Annotator (see the [INSTALL.md](./INSTALL.md) file for instructions on how to install Bystro Annotator). If you don't have Bystro Annotator installed, you can still run the examples as long as you install the Bystro VCF preprocessor by running `go install github.com/bystrogenomics/bystro-vcf@2.2.3`.

Please note that we are not showing the full Bystro Annotator output described below. We're showing just the first 17 columns of the output, which are the most important for understanding the variant representation and sample genotype handling.

```
cat ~/bystro/perl/example_vcf.tsv | bystro-vcf --keepId --emptyField "NA" --keepPos
```

If you also want to output the **Bystro Genotype Dosage Matrix**, at a small performance hit, you can run:

```
cat ~/bystro/perl/example_vcf.tsv | bystro-vcf --keepId --emptyField "NA" --keepPos --dosageOutput example_vcf_dosage_matrix.feather
```

Input Example VCF:

| CHROM | POS     | ID                          | REF  | ALT     | QUAL | FILTER | INFO                              | FORMAT      | NA00001        | NA00002        | NA00003        |
| ----- | ------- | --------------------------- | ---- | ------- | ---- | ------ | --------------------------------- | ----------- | -------------- | -------------- | -------------- |
| 20    | 1       | SIMPLE_SNP                  | A    | T       | 50   | PASS   | .                                 | GT:GQ:DP:HQ | 0/1:54:7:56,60 | 0/0:48:4:51,51 | 0/0:48:4:51,51 |
| 20    | 1110696 | MULTIALLELIC_SNP            | A    | G,T     | 67   | PASS   | NS=2;DP=10;AF=0.333,0.667;AA=T;DB | GT:GQ:DP    | 1/2:21:6       | 2/1:2:0        | 2/2:35:4       |
| 20    | 1       | SIMPLE_INSERTION            | A    | AC      | 50   | PASS   | .                                 | GT:GQ:DP    | 0/0:54:7       | 0/0:48:4       | 1/0:61:2       |
| 20    | 1       | INSERTION_BETWEEN_TWO_BASES | AT   | ACCT    | 50   | PASS   | .                                 | GT:GQ:DP    | 0/1:35:4       | 0/1:17:2       | 1/1:40:3       |
| 20    | 1234567 | microsat1                   | GTCT | G,GTACT | 50   | PASS   | .                                 | GT:GQ:DP    | 0/1:35:4       | 0/2:17:2       | 1/1:40:3       |
| 20    | 3       | EXAMPLE_MISSING_MNP         | CCC  | AAA     | 50   | PASS   | NS=3;DP=9;AA=G                    | GT          | ./1            | 0/0            | 1/1            |

Expected Bystro VCF preprocessor Output:

| chrom | pos     | type         | inputRef | alt | trTv | heterozygotes   | heterozygosity | homozygotes | homozygosity | missingGenos | missingness | ac  | an  | sampleMaf | vcfPos  | id                          |
| ----- | ------- | ------------ | -------- | --- | ---- | --------------- | -------------- | ----------- | ------------ | ------------ | ----------- | --- | --- | --------- | ------- | --------------------------- |
| chr20 | 1       | SNP          | A        | T   | 2    | NA00001         | 0.333          | NA          | 0            | NA           | 0           | 1   | 6   | 0.167     | 1       | SIMPLE_SNP                  |
| chr20 | 1110696 | MULTIALLELIC | A        | G   | 0    | NA00001;NA00002 | 0.667          | NA          | 0            | NA           | 0           | 2   | 6   | 0.333     | 1110696 | MULTIALLELIC_SNP            |
| chr20 | 1110696 | MULTIALLELIC | A        | T   | 0    | NA00001;NA00002 | 0.667          | NA00003     | 0.333        | NA           | 0           | 4   | 6   | 0.667     | 1110696 | MULTIALLELIC_SNP            |
| chr20 | 1       | INS          | A        | +C  | 0    | NA00003         | 0.333          | NA          | 0            | NA           | 0           | 1   | 6   | 0.167     | 1       | SIMPLE_INSERTION            |
| chr20 | 1       | INS          | A        | +CC | 0    | NA00001;NA00002 | 0.667          | NA00003     | 0.333        | NA           | 0           | 4   | 6   | 0.667     | 1       | INSERTION_BETWEEN_TWO_BASES |
| chr20 | 1234568 | MULTIALLELIC | T        | -3  | 0    | NA00001         | 0.333          | NA00003     | 0.333        | NA           | 0           | 3   | 6   | 0.5       | 1234567 | microsat1                   |
| chr20 | 1234568 | MULTIALLELIC | T        | +A  | 0    | NA00002         | 0.333          | NA          | 0            | NA           | 0           | 1   | 6   | 0.167     | 1234567 | microsat1                   |
| chr20 | 3       | MNP          | C        | A   | 2    | NA              | 0              | NA00003     | 0.5          | NA00001      | 0.333       | 2   | 4   | 0.5       | 3       | EXAMPLE_MISSING_MNP         |
| chr20 | 4       | MNP          | C        | A   | 2    | NA              | 0              | NA00003     | 0.5          | NA00001      | 0.333       | 2   | 4   | 0.5       | 3       | EXAMPLE_MISSING_MNP         |
| chr20 | 5       | MNP          | C        | A   | 2    | NA              | 0              | NA00003     | 0.5          | NA00001      | 0.333       | 2   | 4   | 0.5       | 3       | EXAMPLE_MISSING_MNP         |

Expected Bystro Genotype Dosage Matrix Output:

```python
import pandas as pd
df = pd.read_feather('example_vcf_dosage_matrix.feather')
print(df)
```

| locus              | NA0001 | NA0002 | NA0003 |
| ------------------ | ------ | ------ | ------ |
| chr20:1:A:T        | 1      | 0      | 0      |
| chr20:1110696:A:G  | 1      | 1      | 0      |
| chr20:1110696:A:T  | 1      | 1      | 2      |
| chr20:1:A:+C       | 0      | 0      | 1      |
| chr20:1:A:+CC      | 1      | 1      | 2      |
| chr20:1234568:T:-3 | 1      | 0      | 2      |
| chr20:1234568:T:+A | 0      | 1      | 0      |
| chr20:3:C:A        | -1     | 0      | 2      |
| chr20:4:C:A        | -1     | 0      | 2      |
| chr20:5:C:A        | -1     | 0      | 2      |

- Note that missing genotypes are reprsented as -1 in the genotype dosage matrix output
- If a sample's genotype contains any missing genotypes, the sample is considered missing for the site

#### Explanation for SIMPLE_SNP

VCF Representation:

| CHROM | POS | ID         | REF | ALT | QUAL | FILTER | INFO | FORMAT      | NA00001        | NA00002        | NA00003        |
| ----- | --- | ---------- | --- | --- | ---- | ------ | ---- | ----------- | -------------- | -------------- | -------------- |
| 20    | 1   | SIMPLE_SNP | A   | T   | 50   | PASS   | .    | GT:GQ:DP:HQ | 0/1:54:7:56,60 | 0/0:48:4:51,51 | 0/0:48:4:51,51 |

Bystro Representation:

| chrom | pos | type | inputRef | alt | trTv | heterozygotes | heterozygosity | homozygotes | homozygosity | missingGenos | missingness | ac  | an  | sampleMaf | vcfPos | id         |
| ----- | --- | ---- | -------- | --- | ---- | ------------- | -------------- | ----------- | ------------ | ------------ | ----------- | --- | --- | --------- | ------ | ---------- |
| chr20 | 1   | SNP  | A        | T   | 2    | NA00001       | 0.333          | NA          | 0            | NA           | 0           | 1   | 6   | 0.167     | 1      | SIMPLE_SNP |

Bystro Genotype Dosage Matrix Output:

| locus       | NA0001 | NA0002 | NA0003 |
| ----------- | ------ | ------ | ------ |
| chr20:1:A:T | 1      | 0      | 0      |

The Bystro and VCF formats for simple, well-normalized SNPs are the same. In addition to the position, variant type, reference, and alternate, Bystro's VCF preprocessor (bystro-vcf) also outputs whether a variant is a transition (1), transversion (2) or neither (0), descriptive information about the genotypes, including which samples are heterozyogtes, homozygotes, or missing genotypes, vcfPos (which describes the original position in the VCF file, pre-normalization), and the VCF ID. Meanwhile the genotype dosage matrix output shows the number of alternate alleles for each sample at each variant.

#### Explanation for MULTIALLELIC_SNP

VCF Representation:

| CHROM | POS     | ID               | REF | ALT | QUAL | FILTER | INFO                              | FORMAT   | NA00001  | NA00002 | NA00003  |
| ----- | ------- | ---------------- | --- | --- | ---- | ------ | --------------------------------- | -------- | -------- | ------- | -------- |
| 20    | 1110696 | MULTIALLELIC_SNP | A   | G,T | 67   | PASS   | NS=2;DP=10;AF=0.333,0.667;AA=T;DB | GT:GQ:DP | 1/2:21:6 | 2/1:2:0 | 2/2:35:4 |

Bystro Representation:

| chrom | pos     | type         | inputRef | alt | trTv | heterozygotes   | heterozygosity | homozygotes | homozygosity | missingGenos | missingness | ac  | an  | sampleMaf | vcfPos  | id               |
| ----- | ------- | ------------ | -------- | --- | ---- | --------------- | -------------- | ----------- | ------------ | ------------ | ----------- | --- | --- | --------- | ------- | ---------------- |
| chr20 | 1110696 | MULTIALLELIC | A        | G   | 0    | NA00001;NA00002 | 0.667          | NA          | 0            | NA           | 0           | 2   | 6   | 0.333     | 1110696 | MULTIALLELIC_SNP |
| chr20 | 1110696 | MULTIALLELIC | A        | T   | 0    | NA00001;NA00002 | 0.667          | NA00003     | 0.333        | NA           | 0           | 4   | 6   | 0.667     | 1110696 | MULTIALLELIC_SNP |

The VCF representation shows two different SNPs at the same position. NA00001 and NA00002 have 1 copy of each allele, while NA00003 has 2 copies of the A>T allele and 0 copies of A>G. Bystro's representation decomposes the multiallelic site into two separate rows, one for each allele. The first row shows the A>G allele, and the second row shows the A>T allele. Since NA00001 and NA00002 are heterozygous for both A>G and A>T, on each line they are listed in the heterozygotes columns, while NA00003 is homozygous for A>T and is listed in the homozygotes column only for the A>T allele row. The zygosity and sampleMaf (sample minor allele frequency) fields are calculated based on the allele in the row.

#### Explanation for SIMPLE_INSERTION

VCF Representation:

| CHROM | POS | ID               | REF | ALT | QUAL | FILTER | INFO | FORMAT   | NA00001  | NA00002  | NA00003  |
| ----- | --- | ---------------- | --- | --- | ---- | ------ | ---- | -------- | -------- | -------- | -------- |
| 20    | 1   | SIMPLE_INSERTION | A   | AC  | 50   | PASS   | .    | GT:GQ:DP | 0/0:54:7 | 0/0:48:4 | 1/0:61:2 |

Bystro Representation:

| chrom | pos | type | inputRef | alt | trTv | heterozygotes | heterozygosity | homozygotes | homozygosity | missingGenos | missingness | ac  | an  | sampleMaf | vcfPos | id               |
| ----- | --- | ---- | -------- | --- | ---- | ------------- | -------------- | ----------- | ------------ | ------------ | ----------- | --- | --- | --------- | ------ | ---------------- |
| chr20 | 1   | INS  | A        | +C  | 0    | NA00003       | 0.333          | NA          | 0            | NA           | 0           | 1   | 6   | 0.167     | 1      | SIMPLE_INSERTION |

The VCF representation shows an insertion of a C base after the A base at position 1. Bystro's representation shows the insertion as occurring at the A base, with the reference base being A and the alternate allele being +C. The heterozygotes column lists NA00003 as heterozygous for the insertion.

#### Explanation for INSERTION_BETWEEN_TWO_BASES

VCF Representation:

| CHROM | POS | ID                          | REF | ALT  | QUAL | FILTER | INFO | FORMAT   | NA00001  | NA00002  | NA00003  |
| ----- | --- | --------------------------- | --- | ---- | ---- | ------ | ---- | -------- | -------- | -------- | -------- |
| 20    | 1   | INSERTION_BETWEEN_TWO_BASES | AT  | ACCT | 50   | PASS   | .    | GT:GQ:DP | 0/1:35:4 | 0/1:17:2 | 1/1:40:3 |

Bystro Representation:

| chrom | pos | type | inputRef | alt | trTv | heterozygotes   | heterozygosity | homozygotes | homozygosity | missingGenos | missingness | ac  | an  | sampleMaf | vcfPos | id                          |
| ----- | --- | ---- | -------- | --- | ---- | --------------- | -------------- | ----------- | ------------ | ------------ | ----------- | --- | --- | --------- | ------ | --------------------------- |
| chr20 | 1   | INS  | A        | +CC | 0    | NA00001;NA00002 | 0.667          | NA00003     | 0.333        | NA           | 0           | 4   | 6   | 0.667     | 1      | INSERTION_BETWEEN_TWO_BASES |

The VCF representation shows an insertion of CC between the A and T bases. Bystro's representation shows the insertion as occurring after the A base, with the reference base being A and the alternate allele being +CC. NA00001 and NA00002 are heterozygous for the insertion, while NA00003 is homozygous for the insertion and therefore listed in the homozygotes column.

#### Explanation for microsat1

VCF Representation:

| CHROM | POS     | ID        | REF  | ALT     | QUAL | FILTER | INFO | FORMAT   | NA00001  | NA00002  | NA00003  |
| ----- | ------- | --------- | ---- | ------- | ---- | ------ | ---- | -------- | -------- | -------- | -------- |
| 20    | 1234567 | microsat1 | GTCT | G,GTACT | 50   | PASS   | .    | GT:GQ:DP | 0/1:35:4 | 0/2:17:2 | 1/1:40:3 |

Bystro Representation:

| chrom | pos     | type         | inputRef | alt | trTv | heterozygotes | heterozygosity | homozygotes | homozygosity | missingGenos | missingness | ac  | an  | sampleMaf | vcfPos  | id        |
| ----- | ------- | ------------ | -------- | --- | ---- | ------------- | -------------- | ----------- | ------------ | ------------ | ----------- | --- | --- | --------- | ------- | --------- |
| chr20 | 1234568 | MULTIALLELIC | T        | -3  | 0    | NA00001       | 0.333          | NA00003     | 0.333        | NA           | 0           | 3   | 6   | 0.5       | 1234567 | microsat1 |
| chr20 | 1234568 | MULTIALLELIC | T        | +A  | 0    | NA00002       | 0.333          | NA          | 0            | NA           | 0           | 1   | 6   | 0.167     | 1234567 | microsat1 |

The VCF representation shows a multiallelic site with two alleles. The first allele is GTCT>G at position 124567, because the VCF format's POS is the first base of the reference. In reality, this deletion is the deletion of the TCT bases starting at position 1234568, but because of VCF's padding requirements, the VCF format cannot show it as such. Bystro shows this allele at the correct position, 1234568, as a `-3`. 2 samples have this allele, NA00001 and NA00003. NA00001 is heterozygous, and NA00003 is homozygous, and are listed as such in the Bystro output.

The second allele is GTCT>GTACT, with the insertion of an "A" occuring after the "T" base at position 1234568. Again, because of the VCF format's padding rule, this representation cannot be shown directly in the VCF format, but must be inferred. Bystro normalizes the representation, showing the insertion at the correct base, 1234568.

#### Explanation for EXAMPLE_MISSING_MNP

VCF Representation:

| CHROM | POS | ID                  | REF | ALT | QUAL | FILTER | INFO           | FORMAT | NA00001 | NA00002 | NA00003 |
| ----- | --- | ------------------- | --- | --- | ---- | ------ | -------------- | ------ | ------- | ------- | ------- |
| 20    | 3   | EXAMPLE_MISSING_MNP | CCC | AAA | 50   | PASS   | NS=3;DP=9;AA=G | GT     | ./1     | 0/0     | 1/1     |

Bystro Representation:

| chrom | pos | type | inputRef | alt | trTv | heterozygotes | heterozygosity | homozygotes | homozygosity | missingGenos | missingness | ac  | an  | sampleMaf | vcfPos | id                  |
| ----- | --- | ---- | -------- | --- | ---- | ------------- | -------------- | ----------- | ------------ | ------------ | ----------- | --- | --- | --------- | ------ | ------------------- |
| chr20 | 3   | MNP  | C        | A   | 2    | NA            | 0              | NA00003     | 0.5          | NA00001      | 0.333       | 2   | 4   | 0.5       | 3      | EXAMPLE_MISSING_MNP |
| chr20 | 4   | MNP  | C        | A   | 2    | NA            | 0              | NA00003     | 0.5          | NA00001      | 0.333       | 2   | 4   | 0.5       | 3      | EXAMPLE_MISSING_MNP |
| chr20 | 5   | MNP  | C        | A   | 2    | NA            | 0              | NA00003     | 0.5          | NA00001      | 0.333       | 2   | 4   | 0.5       | 3      | EXAMPLE_MISSING_MNP |

The VCF representation shows a multi-nucleotide polymorphism (MNP) at position 3, where 3 bases are changed from CCC to AAA. An MNP is really 3 single nucleotide polymorphisms next to each other, typically linked on the same chromosome. Bystro decomposes the MNP into 3 separate rows, each with a single nucleotide change. The first row shows the first base change, the second row shows the second base change, and the third row shows the third base change. NA00001 was unsuccessfully typed, with 1 of its 2 chromosomes having an ambiguous or low quality genotype ("."). Bystro, to be conservative ("garbage in means garbage out"), counts this sample as having a missing genotype, and subtracts 2 from the `an` (allele number).

## What is the Bystro Annotation Database?

To output annotations, the user must point Bystro Annotator at a Bystro Annotator Database, which is a high-performance embedded memory-mapped database used by the Bystro Annotator to label variants. Three default databases are provided, for Humans (hg19 and hg38), and rats (rn7). See the [INSTALL.md#databases](./INSTALL.md#databases) section for more information on how to download these databases.

Key points:

- A Bystro Annotation Database is a high-performance memory-mapped key-value database that uses the Lightning Memory Map Database (LMDB) engine. It supports millions of lookups per second on a single machine, and can be used to store and retrieve annotations for millions of variants.

- Bystro Annotation Databases can be re-created from the YAML configuration file corresponding to that database, and new databases with different information can be created by editing the YAML configuration file, and re-running the Bystro Annotation Database creation process.

- To create a Bystro Annotation Database, the user needs to provide a YAML configuration file that specifies all of the source file locations, the location to write the database, and the tracks/fields to output, and then runs `bystro-build.pl --config /path/to/config`. This will create a Bystro Annotation Database that can be used to annotate VCF or SNP files.

## Annotation Fields for Default Human Assembly hg38 and hg19 Bystro Annotation Databases

### Basic Fields

Sourced from the input file, or calculated based on input fields from the VCF or SNP file pre-processor.

`chrom` - chromosome, always prepended with "chr"

`pos` - genomic position after Bystro normalizes variant representations

- Positions always correspond to the first affected base.

`type` - the type of variant

- VCF format types: `SNP`, `INS`, `DEL`, `MULTIALLELIC`
  - Multi-nucleotide polymorphisms are decomposed into separate rows of `type` "SNP", each of 1 variant. In the next release of Bystro these will be called "MNP" and have a "linkage" property to enable you to reconstruct the original MNP, while still retaining the full set of per-SNP annotations
  - Multiallelics are decomposed into separate rows, but retain the "MULTIALLELIC" `type`
- SNP format types: `SNP`, `INS`, `DEL`, `MULTIALLELIC`, `DENOVO`

`inputRef` - the reference base, as it is present in the input file, after variant and reference normalization.

- This is generated by the input file pre-processor (hence `input` in the name), and is always 1 base long - the affected reference base at that position

`alt` - the alternate/nonreference allele

- VCF multi-allelic and MNP sites are decomposed into individual entries of a single allele.
  - Genotypes are properly segregated per allele

`trTv` - transition:transversion ratio for your dataset at this position

- Possible values: `0`, `1`, `2`
  - 0 indicates neither transition nor transversion (which occurs when the alternate allele is an insertion or deleetion)
  - 1 is a transition (purine -> purine or pyrimidine -> pyrimidine)
  - 2 is a transversion (pyridine -> pyrimidine or vice versa)

`heterozygotes` - The array of heterozygous sample labels

`heterozygosity` - The fraction of all samples (excluding missing samples) that are heterozygous for the alternate allele

`homozygotes` - The array of homozygous sample labels

`homozygosity` - The fraction of all samples that are homozygous for the alternate allele

`missingGenos` - The samples that did not have a genotype (e.g., ".") at the site. If an individual has at least 1 missing genotype, they are considered missing for the site.

- e.g., .|., .|0 are all considered missing

`missingness` - The fraction of all samples that have missing genotypes for the alternate allele

`ac` - The alternate allele count

`an` - The total non-mising allele count

`sampleMaf` - The in-sample alternate allele frequency

`vcfPos` - The original VCF `POS`, unaffected by Bystro normalization transformations

`id` - The VCF `ID` field, if any

`discordant` - TRUE if the reference base provided in the input VCF matches the Bystro-annotated UCSC reference, FALSE otherwise

`ref` - The Bystro-annotated reference base(s), from the 'ref' track in the Bystro Annotation Database

- In the default Bystro Annotation Database, this is sourced from the UCSC reference genome
- In custom Bystro Annotation Databases, this can be sourced from any reference genome
- In the case of insertions the `ref` will be 2 bases long, the base just before the insertion, and the one right after
- In the case of deletions, the ref will be as long as the deletion, up to 32 bases (after that, the ref will be truncated)

<br/>

### Transcript Annotations

In the default Bystro Annotation Database, we source transcript annotations from the UCSC refGene track, joined on other UCSC tracks: knownToEnsembl, kgXref, knownCanonical.

- See [refGene](https://genome.ucsc.edu/cgi-bin/hgTables?db=hg38&hgta_group=genes&hgta_track=refSeqComposite&hgta_table=refGene&hgta_doSchema=describe+table+schema) and [kgXref](https://genome.cse.ucsc.edu/cgi-bin/hgTables?hgsid=1893397768_GDljX7p8FQaqUVJ3FZD1cSUFpeV2&hgta_doSchemaDb=hg38&hgta_doSchemaTable=kgXref) for more information

- In custom Bystro Annotation Databases, these annotations can be sourced from any UCSC transcript track, and multiple such `gene` type tracks can be defined in a single Bystro Annotation Database (annotations for all will be outputted)

- **When a site is intergenic, all `refSeq` annotations will be `NA`**

`refSeq.siteType` - the kind of effect the `alt` allele has on this transcript.

- Possible types: `intronic`, `exonic`, `UTR3`, `UTR5`, `spliceAcceptor`, `spliceDonor`, `ncRNA`

`refSeq.exonicAlleleFunction` - The coding effect of the variant

- Possible values: `synonymous`, `nonSynonymous`, `indel-nonFrameshift`, `indel-frameshift`, `stopGain`, `stopLoss`, `startLoss`
- This will be `NA `for non-coding `siteType`

`refSeq.refCodon`- The reference codon based on _in silico_ transcription of the reference assembly

`refSeq.altCodon` - The _in silico_ transcribed codon after modification by the `alt` allele

`refSeq.refAminoAcid`- The amino acid based on _in silico_ translation of the reference transcript

`refSeq.altAminoAcid` - The _in-silico_ translated amino acid after modification by the `alt` allele

`refSeq.codonPosition` - The site's position within the codon (1, 2, 3)

`refSeq.codonNumber` - The codon number within the transcript

`refSeq.strand` - The positive or negative (Watson or Crick) strand

`refSeq.name` - RefSeq transcript ID

`refSeq.name2` - RefSeq gene sybmol

`refSeq.description` - The long form description of the RefSeq transcript

`refSeq.kgID` - UCSC's <a href='https://www.ncbi.nlm.nih.gov/pubmed/16500937' target='_blank'>Known Genes</a> ID

`refSeq.mRNA` - The mRNA ID, the transcript ID starting with NM\_

`refSeq.spID` - <a href="http://www.uniprot.org" target='_blank'>UniProt</a> protein accession number

`refSeq.spDisplayID` - <a href="http://www.uniprot.org" target='_blank'>UniProt</a> display ID

`refSeq.protAcc` - NCBI protein accession number

`refSeq.rfamAcc` - <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC165453/" target="_blank">Rfam</a> accession number

`refSeq.tRnaName` - Name from the tRNA track

`refSeq.ensemblID` - The Ensembl transcript id

`refSeq.isCanonical` - Whether the transcript is the canonical splice variant for the gene

<br/>

### nearest.refSeq

The nearest transcript(s), calculated by trascript start, transcript end boundaries. Transcripts that are equidistant are all outputted.

`nearest.refSeq.name` - The nearest transcript(s) RefSeq transcript ID

`nearest.refSeq.name2` - The nearest transcript(s) RefSeq gene symbol

`nearest.refSeq.dist` - The distance to these transcripts. Negative values indicate the site is downstream of the transcript

<br/>

### nearestTss.refSeq

The nearest transcript(s), calculated by the distance to the nearest transcript start site (TSS). Transcripts with the same TSS are all outputted.

`nearestTss.refSeq.name` - The nearest transcript(s) RefSeq transcript ID

`nearestTss.refSeq.name2` - The nearest transcript(s) RefSeq gene symbol

`nearestTss.refSeq.dist` - A single value; the distance to these transcripts' transcription start site. Negative values indicate the site is downstream of the TSS

<br/>

### gnomAD Annotations

Annotations from the gnomAD v4.1 (hg38 assembly annotations) or v2.1.1 (hg19 assembly annotations) whole-genome set

Since the data available for hg19 and hg38 differ, we will discuss them separately below.

<br/>

### hg38 gnomad.joint

Annotations from the gnomAD v4.1 (hg38 assembly annotations) joint set

`gnomad.joint.alt`: The Bystro VCF-preprocessor's ALT record for this gnomAD site. This should always match the row's `alt` field value

`gnomad.joint.id`: The VCF `ID` field

`gnomad.joint.AF_exomes`: Alternate allele frequency in exomes

`gnomad.joint.AN_exomes`: Total number of alleles in exomes

`gnomad.joint.AF_genomes`: Alternate allele frequency in genomes

`gnomad.joint.AF_joint`: Alternate allele frequency in joint subset

`gnomad.joint.AN_joint`: Total number of alleles in joint subset

`gnomad.joint.AN_genomes`: Total number of alleles in genomes

`gnomad.joint.AF_joint_XX`: Alternate allele frequency in XX samples in joint subset

`gnomad.joint.AN_joint_XX`: Total number of alleles in XX samples in joint subset

`gnomad.joint.AF_joint_XY`: Alternate allele frequency in XY samples in joint subset

`gnomad.joint.AN_joint_XY`: Total number of alleles in XY samples in joint subset

`gnomad.joint.AF_joint_afr`:Alternate allele frequency in samples of African/African-American ancestry in joint subset

`gnomad.joint.AN_joint_afr`: Total number of alleles in samples of African/African-American ancestry in joint subset

`gnomad.joint.AF_joint_ami`: Alternate allele frequency in samples of Amish ancestry in joint subset

`gnomad.joint.AN_joint_ami`: Total number of alleles in samples of Amish ancestry in joint subset

`gnomad.joint.AF_joint_amr`: Alternate allele frequency in samples of Latino ancestry in joint subset

`gnomad.joint.AN_joint_amr`: Total number of alleles in samples of Latino ancestry in joint subset

`gnomad.joint.AF_joint_asj`: Alternate allele frequency in samples of Ashkenazi Jewish ancestry in joint subset

`gnomad.joint.AN_joint_asj`: Total number of alleles in samples of Ashkenazi Jewish ancestry in joint subset

`gnomad.joint.AF_joint_eas`: Alternate allele frequency in samples of East Asian ancestry in joint subset

`gnomad.joint.AN_joint_eas`: Total number of alleles in samples of East Asian ancestry in joint subset

`gnomad.joint.AF_joint_fin`: Alternate allele frequency in samples of Finnish ancestry in joint subset

`gnomad.joint.AN_joint_fin`: Total number of alleles in samples of Finnish ancestry in joint subset

`gnomad.joint.AF_joint_mid`: Alternate allele frequency in samples of Middle Eastern ancestry in joint subset

`gnomad.joint.AN_joint_mid`: Total number of alleles in samples of Middle Eastern ancestry in joint subset

`gnomad.joint.AF_joint_nfe`: Alternate allele frequency in samples of Non-Finnish European ancestry in joint subset

`gnomad.joint.AN_joint_nfe`: Total number of alleles in samples of Non-Finnish European ancestry in joint subset

`gnomad.joint.AF_joint_raw`: Alternate allele frequency in samples, before removing low-confidence genotypes in joint dataset

`gnomad.joint.AN_joint_raw`: Total number of alleles in samples, before removing low-confidence genotypes in joint dataset

`gnomad.joint.AF_joint_remaining`: Alternate allele frequency in samples in the Remaining individuals genetic ancestry group in joint dataset

`gnomad.joint.AN_joint_remaining`: Total number of alleles in samples in the Remaining individuals genetic ancestry group in joint dataset

`gnomad.joint.AF_joint_sas`: Alternate allele frequency in samples in the South Asian genetic ancestry group in joint dataset

`gnomad.joint.AN_joint_sas`: Total number of alleles in samples in the South Asian genetic ancestry group in joint dataset

`gnomad.joint.AF_grpmax_joint`: Maximum allele frequency across genetic ancestry groups in the joint subset

`gnomad.joint.AN_grpmax_joint`: Total number of alleles in the genetic ancestry group with the maximum allele frequency in the joint subset

<br/>

### hg38 gnomad.genomes

Annotations from the gnomAD v4.1 whole-genome set

`gnomad.genomes.alt`: The Bystro VCF-preprocessor's ALT record for this gnomAD site. This should always match the row's `alt` field value

`gnomad.genomes.id`: The VCF `ID` field

`gnomad.genomes.spliceai_ds_max`: Illumina's SpliceAI max delta score; interpreted as the probability of the variant being splice-altering

`gnomad.genomes.pangolin_largest_ds`: Pangolin's largest delta score across 2 splicing consequences, which reflects the probability of the variant being splice-altering

`gnomad.genomes.phylop`: Base-wise conservation score across the 241 placental mammals in the Zoonomia project. Score ranges from -20 to 9.28, and reflects acceleration (faster evolution than expected under neutral drift, assigned negative scores)

`gnomad.genomes.sift_max`: Score reflecting the scaled probability of the amino acid substitution being tolerated, ranging from 0 to 1. Scores below 0.05 are predicted to impact protein function. We prioritize max scores for MANE Select transcripts where possible and otherwise report a score for the canonical transcript

`gnomad.genomes.polyphen_max`: Score that predicts the possible impact of an amino acid substitution on the structure and function of a human protein, ranging from 0.0 (tolerated) to 1.0 (deleterious). We prioritize max scores for MANE Select transcripts where possible and otherwise report a score for the canonical transcript

`gnomad.genomes.AN`: Total number of alleles

`gnomad.genomes.AF`: Alternate allele frequency

`gnomad.genomes.AF_XX`: Alternate allele frequency in XX samples

`gnomad.genomes.AN_XX`: Total number of alleles in XX samples

`gnomad.genomes.AF_XY`: Alternate allele frequency in XY samples

`gnomad.genomes.AN_XY`: Total number of alleles in XY samples

`gnomad.genomes.AF_afr`: Alternate allele frequency in samples of African/African-American ancestry

`gnomad.genomes.AN_afr`: Total number of alleles in samples of African/African-American ancestry

`gnomad.genomes.AF_ami`: Alternate allele frequency in samples of Amish ancestry

`gnomad.genomes.AN_ami`: Total number of alleles in samples of Amish ancestry

`gnomad.genomes.AF_amr`: Alternate allele frequency in samples of Latino ancestry

`gnomad.genomes.AN_amr`: Total number of alleles in samples of Latino ancestry

`gnomad.genomes.AF_asj`: Alternate allele frequency in samples of Ashkenazi Jewish ancestry

`gnomad.genomes.AN_asj`: Total number of alleles in samples of Ashkenazi Jewish ancestry

`gnomad.genomes.AF_eas`: Alternate allele frequency in samples of East Asian ancestry

`gnomad.genomes.AN_eas`: Total number of alleles in samples of East Asian ancestry

`gnomad.genomes.AF_fin`: Alternate allele frequency in samples of Finnish ancestry

`gnomad.genomes.AN_fin`: Total number of alleles in samples of Finnish ancestry

`gnomad.genomes.AF_mid`: Alternate allele frequency in samples of Middle Eastern ancestry

`gnomad.genomes.AN_mid`: Total number of alleles in samples of Middle Eastern ancestry

`gnomad.genomes.AF_nfe`: Alternate allele frequency in samples of Non-Finnish European ancestry

`gnomad.genomes.AN_nfe`: Total number of alleles in samples of Non-Finnish European ancestry

`gnomad.genomes.AF_remaining`: Alternate allele frequency in samples of Remaining individuals ancestry

`gnomad.genomes.AN_remaining`: Total number of alleles in samples of Remaining individuals ancestry

`gnomad.genomes.AF_sas`: Alternate allele frequency in samples of South Asian ancestry

`gnomad.genomes.AN_sas`: Total number of alleles in samples of South Asian ancestry

`gnomad.genomes.AF_grpmax`: Maximum allele frequency across genetic ancestry groups

`gnomad.genomes.AN_grpmax`: Total number of alleles in the genetic ancestry group with the maximum allele frequency

<br/>

### hg38 gnomad.exomes

Annotations from gnomAD v4.1 whole-exome set

`gnomad.exomes.alt`: The Bystro VCF-preprocessor's ALT record for this gnomAD site. This should always match the row's `alt` field value

`gnomad.exomes.id`: The VCF `ID` field

`gnomad.exomes.spliceai_ds_max`: Illumina's SpliceAI max delta score; interpreted as the probability of the variant being splice-altering

`gnomad.exomes.pangolin_largest_ds`: Pangolin's largest delta score across 2 splicing consequences, which reflects the probability of the variant being splice-altering

`gnomad.exomes.phylop`: Base-wise conservation score across the 241 placental mammals in the Zoonomia project. Score ranges from -20 to 9.28, and reflects acceleration (faster evolution than expected under neutral drift, assigned negative scores)

`gnomad.exomes.sift_max`: Score reflecting the scaled probability of the amino acid substitution being tolerated, ranging from 0 to 1. Scores below 0.05 are predicted to impact protein function. We prioritize max scores for MANE Select transcripts where possible and otherwise report a score for the canonical transcript

`gnomad.exomes.polyphen_max`: Score that predicts the possible impact of an amino acid substitution on the structure and function of a human protein, ranging from 0.0 (tolerated) to 1.0 (deleterious). We prioritize max scores for MANE Select transcripts where possible and otherwise report a score for the canonical transcript

`gnomad.exomes.AN`: Total number of alleles

`gnomad.exomes.AF`: Alternate allele frequency

`gnomad.exomes.AF_XX`: Alternate allele frequency in XX samples

`gnomad.exomes.AN_XX`: Total number of alleles in XX samples

`gnomad.exomes.AF_XY`: Alternate allele frequency in XY samples

`gnomad.exomes.AN_XY`: Total number of alleles in XY samples

`gnomad.exomes.AF_afr`: Alternate allele frequency in samples of African/African-American ancestry

`gnomad.exomes.AN_afr`: Total number of alleles in samples of African/African-American ancestry

`gnomad.exomes.AF_amr`: Alternate allele frequency in samples of Latino ancestry

`gnomad.exomes.AN_amr`: Total number of alleles in samples of Latino ancestry

`gnomad.exomes.AF_asj`: Alternate allele frequency in samples of Ashkenazi Jewish ancestry

`gnomad.exomes.AN_asj`: Total number of alleles in samples of Ashkenazi Jewish ancestry

`gnomad.exomes.AF_eas`: Alternate allele frequency in samples of East Asian ancestry

`gnomad.exomes.AN_eas`: Total number of alleles in samples of East Asian ancestry

`gnomad.exomes.AF_fin`: Alternate allele frequency in samples of Finnish ancestry

`gnomad.exomes.AN_fin`: Total number of alleles in samples of Finnish ancestry

`gnomad.exomes.AF_mid`: Alternate allele frequency in samples of Middle Eastern ancestry

`gnomad.exomes.AN_mid`: Total number of alleles in samples of Middle Eastern ancestry

`gnomad.exomes.AF_nfe`: Alternate allele frequency in samples of Non-Finnish European ancestry

`gnomad.exomes.AN_nfe`: Total number of alleles in samples of Non-Finnish European ancestry

`gnomad.exomes.AF_non_ukb`: Alternate allele frequency in non_ukb subset

`gnomad.exomes.AN_non_ukb`: Total number of alleles in non_ukb subset

`gnomad.exomes.AF_non_ukb_afr`: Alternate allele frequency in samples of African/African-American ancestry in non_ukb subset

`gnomad.exomes.AN_non_ukb_afr`: Total number of alleles in samples of African/African-American ancestry in non_ukb subset

`gnomad.exomes.AF_non_ukb_amr`: Alternate allele frequency in samples of Latino ancestry in non_ukb subset

`gnomad.exomes.AN_non_ukb_amr`: Total number of alleles in samples of Latino ancestry in non_ukb subset

`gnomad.exomes.AF_non_ukb_asj`: Alternate allele frequency in samples of Ashkenazi Jewish ancestry in non_ukb subset

`gnomad.exomes.AN_non_ukb_asj`: Total number of alleles in samples of Ashkenazi Jewish ancestry in non_ukb subset

`gnomad.exomes.AF_non_ukb_eas`: Alternate allele frequency in samples of East Asian ancestry in non_ukb subset

`gnomad.exomes.AN_non_ukb_eas`: Total number of alleles in samples of East Asian ancestry in non_ukb subset

`gnomad.exomes.AF_non_ukb_fin`: Alternate allele frequency in samples of Finnish ancestry in non_ukb subset

`gnomad.exomes.AN_non_ukb_fin`: Total number of alleles in samples of Finnish ancestry in non_ukb subset

`gnomad.exomes.AF_non_ukb_mid`: Alternate allele frequency in samples of Middle Eastern ancestry in non_ukb subset

`gnomad.exomes.AN_non_ukb_mid`: Total number of alleles in samples of Middle Eastern ancestry in non_ukb subset

`gnomad.exomes.AF_non_ukb_nfe`: Alternate allele frequency in samples of Non-Finnish European ancestry in non_ukb subset

`gnomad.exomes.AN_non_ukb_nfe`: Total number of alleles in samples of Non-Finnish European ancestry in non_ukb subset

`gnomad.exomes.AF_non_ukb_remaining`: Alternate allele frequency in samples of Remaining individuals ancestry in non_ukb subset

`gnomad.exomes.AN_non_ukb_remaining`: Total number of alleles in samples of Remaining individuals ancestry in non_ukb subset

`gnomad.exomes.AF_non_ukb_sas`: Alternate allele frequency in samples of South Asian ancestry in non_ukb subset

`gnomad.exomes.AN_non_ukb_sas`: Total number of alleles in samples of South Asian ancestry in non_ukb subset

`gnomad.exomes.AF_remaining`: Alternate allele frequency in samples of Remaining individuals ancestry

`gnomad.exomes.AN_remaining`: Total number of alleles in samples of Remaining individuals ancestry

`gnomad.exomes.AF_sas`: Alternate allele frequency in samples of South Asian ancestry

`gnomad.exomes.AN_sas`: Total number of alleles in samples of South Asian ancestry

`gnomad.exomes.AF_grpmax`: Maximum allele frequency across genetic ancestry groups

`gnomad.exomes.AN_grpmax`: Total number of alleles in the genetic ancestry group with the maximum allele frequency

`gnomad.exomes.AF_grpmax_non_ukb`: Maximum allele frequency across genetic ancestry groups in non_ukb subset

`gnomad.exomes.AN_grpmax_non_ukb`: Total number of alleles in the genetic ancestry group with the maximum allele frequency in non_ukb subset

`gnomad.exomes.AF_grpmax_joint`: Maximum allele frequency across genetic ancestry groups in joint subset

`gnomad.exomes.AN_grpmax_joint`: Total number of alleles in the genetic ancestry group with the maximum allele frequency in joint subset

<br/>

### hg19 gnomad.genomes (v2.1.1 - latest release for hg19)

`gnomad.genomes.alt`: The Bystro VCF-preprocessor's ALT record for this gnomAD site. This should always match the row's `alt` field value

`gnomad.genomes.id`: The VCF `ID` field

`gnomad.genomes.AN`: Total number of alleles

`gnomad.genomes.AF`: Alternate allele frequency

`gnomad.genomes.AN_female`: Total number of alleles in female samples

`gnomad.genomes.AF_female`: Alternate allele frequency in female samples

`gnomad.genomes.non_neuro_AN`: Total number of alleles in samples in the non_neuro subset

`gnomad.genomes.non_neuro_AF`: Alternate allele frequency in samples in the non_neuro subset

`gnomad.genomes.non_topmed_AN`: Total number of alleles in samples in the non_topmed subset

`gnomad.genomes.non_topmed_AF`: Alternate allele frequency in samples in the non_topmed subset

`gnomad.genomes.controls_AN`: Total number of alleles in samples in the controls subset

`gnomad.genomes.controls_AF`: Alternate allele frequency in samples in the controls subset

`gnomad.genomes.AN_nfe_seu`: Total number of alleles in samples of Southern European ancestry

`gnomad.genomes.AF_nfe_seu`: Alternate allele frequency in samples of Southern European ancestry

`gnomad.genomes.AN_nfe_onf`: Total number of alleles in samples of Other Non-Finnish European ancestry

`gnomad.genomes.AF_nfe_onf`: Alternate allele frequency in samples of Other Non-Finnish European ancestry

`gnomad.genomes.AN_amr`: Total number of alleles in samples of Latino ancestry

`gnomad.genomes.AF_amr`: Alternate allele frequency in samples of Latino ancestry

`gnomad.genomes.AN_eas`: Total number of alleles in samples of East Asian ancestry

`gnomad.genomes.AF_eas`: Alternate allele frequency in samples of East Asian ancestry

`gnomad.genomes.AN_nfe_nwe`: Total number of alleles in samples of Northwestern European ancestry

`gnomad.genomes.AF_nfe_nwe`: Alternate allele frequency in samples of Northwestern European ancestry

`gnomad.genomes.AN_nfe_est`: Total number of alleles in samples of Estonian ancestry

`gnomad.genomes.AF_nfe_est`: Alternate allele frequency in samples of Estonian ancestry

`gnomad.genomes.AN_nfe`: Total number of alleles in samples of Non-Finnish European ancestry

`gnomad.genomes.AF_nfe`: Alternate allele frequency in samples of Non-Finnish European ancestry

`gnomad.genomes.AN_fin`: Total number of alleles in samples of Finnish ancestry

`gnomad.genomes.AF_fin`: Alternate allele frequency in samples of Finnish ancestry

`gnomad.genomes.AN_asj`: Total number of alleles in samples of Ashkenazi Jewish ancestry

`gnomad.genomes.AF_asj`: Alternate allele frequency in samples of Ashkenazi Jewish ancestry

`gnomad.genomes.AN_oth`: Total number of alleles in samples of Other ancestry

`gnomad.genomes.AF_oth`: Alternate allele frequency in samples of Other ancestry

<br/>

### hg19 gnomad.exomes (v2.1.1 - latest release for hg19)

Annotations from the gnomAD v2.1.1 exome set

`gnomad.exomes.alt`: The Bystro VCF-preprocessor's ALT record for this gnomAD site. This should always match the row's `alt` field value

`gnomad.exomes.id`: The VCF `ID` field

`gnomad.exomes.AN`: Total number of alleles

`gnomad.exomes.AF`: Alternate allele frequency

`gnomad.exomes.AN_female`: Total number of alleles in female samples

`gnomad.exomes.AF_female`: Alternate allele frequency in female samples

`gnomad.exomes.non_cancer_AN`: Total number of alleles in samples in the non_cancer subset

`gnomad.exomes.non_cancer_AF`: Alternate allele frequency in samples in the non_cancer subset

`gnomad.exomes.non_neuro_AN`: Total number of alleles in samples in the non_neuro subset

`gnomad.exomes.non_neuro_AF`: Alternate allele frequency in samples in the non_neuro subset

`gnomad.exomes.non_topmed_AN`: Total number of alleles in samples in the non_topmed subset

`gnomad.exomes.non_topmed_AF`: Alternate allele frequency in samples in the non_topmed subset

`gnomad.exomes.controls_AN`: Total number of alleles in samples in the controls subset

`gnomad.exomes.controls_AF`: Alternate allele frequency in samples in the controls subset

`gnomad.exomes.AN_nfe_seu`: Total number of alleles in samples of Southern European ancestry

`gnomad.exomes.AF_nfe_seu`: Alternate allele frequency in samples Southern European ancestry

`gnomad.exomes.AN_nfe_bgr`: Total number of alleles in samples of Bulgarian (Eastern European) ancestry

`gnomad.exomes.AF_nfe_bgr`: Alternate allele frequency in samples of Bulgarian (Eastern European) ancestry

`gnomad.exomes.AN_afr`: Total number of alleles in samples of African/African-American ancestry

`gnomad.exomes.AF_afr`: Alternate allele frequency in samples of African/African-American ancestry

`gnomad.exomes.AN_sas`: Total number of alleles in samples of South Asian ancestry

`gnomad.exomes.AF_sas`: Alternate allele frequency in samples of South Asian ancestry

`gnomad.exomes.AN_nfe_onf`: Total number of alleles in samples of Other Non-Finnish European ancestry

`gnomad.exomes.AF_nfe_onf`: Alternate allele frequency in samples of Other Non-Finnish European ancestry

`gnomad.exomes.AN_amr`: Total number of alleles in samples of Latino ancestry

`gnomad.exomes.AF_amr`: Alternate allele frequency in samples of Latino ancestry

`gnomad.exomes.AN_eas`: Total number of alleles in samples of East Asian ancestry

`gnomad.exomes.AF_eas`: Alternate allele frequency in samples of East Asian ancestry

`gnomad.exomes.AN_nfe_swe`: Total number of alleles in samples of Swedish ancestry

`gnomad.exomes.AF_nfe_swe`: Alternate allele frequency in samples of Swedish ancestry

`gnomad.exomes.AN_nfe_nwe`: Total number of alleles in samples of Northwestern European ancestry

`gnomad.exomes.AF_nfe_nwe`: Alternate allele frequency in samples of Northwestern European ancestry

`gnomad.exomes.AN_eas_jpn`: Total number of alleles in samples of Japanese ancestry

`gnomad.exomes.AF_eas_jpn`: Alternate allele frequency in samples of Japanese ancestry

`gnomad.exomes.AN_eas_kor`: Total number of alleles in samples of Korean ancestry

`gnomad.exomes.AF_eas_kor`: Alternate allele frequency in samples of Korean ancestry

<br/>

### [dbSNP](https://www.ncbi.nlm.nih.gov/snp)

dbSNP 155 annotations. Descriptions taken from UCSC's [reference on dbSNP155](https://genome.ucsc.edu/cgi-bin/hgTrackUi?db=hg38&g=dbSnp155Composite)

`dbSNP.id`: The dbSN VCF `ID`

`dbSNP.alt`: The Bystro VCF-preprocessor's ALT record for this dbSNP site. This should always match the row's `alt` field value

`dbSNP.TOMMO`: Allele frequency from the Tohoku Medical Megabank Project contains an allele frequency panel of 3552 Japanese individuals, including the X chromosome

`dbSNP.ExAC`: Allele frequency from the Exome Aggregation Consortium (ExAC) dataset contains 60,706 unrelated individuals sequenced as part of various disease-specific and population genetic studies. Individuals affected by severe pediatric disease have been removed

`dbSNP.GnomAD`: Allele frequency from the gnomAD v3 project. This gnomAD genome dataset includes a catalog containing 602M SNVs and 105M indels based on the whole-genome sequencing of 71,702 samples mapped to the GRCh38 build of the human reference genome.

`dbSNP.Korea1K`: Allele frequency from the Korea1K dataset, which contains 1,094 Korean personal genomes with clinical information

`dbSNP.GoNL`: Allele frequency from the Genome of the Netherlands (GoNL) project. The Genome of the Netherlands (GoNL) Project characterizes DNA sequence variation, common and rare, for SNVs and short insertions and deletions (indels) and large deletions in 769 individuals of Dutch ancestry selected from five biobanks under the auspices of the Dutch hub of the Biobanking and Biomolecular Research Infrastructure (BBMRI-NL).

`dbSNP.KOREAN`: Allele frequency from the Korean Reference Genome Database contains data for 1,465 Korean individuals

`dbSNP.TWINSUK`: Allele frequency from the TwinsUK project. The UK10K - TwinsUK project contains 1854 samples from the Department of Twin Research and Genetic Epidemiology (DTR). The dataset contains data obtained from the 11,000 identical and non-identical twins between the ages of 16 and 85 years old.

`dbSNP.Vietnamese`: Allele frequency from the Kinh Vietnamese database contains 24.81 million variants (22.47 million single nucleotide polymorphisms (SNPs) and 2.34 million indels), of which 0.71 million variants are novel

`dbSNP.GENOME_DK`: Allele frequency from the Danish reference pan genome [phase II](https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJEB19794). The dataset contains the sequencing of Danish parent-offspring trios to determine genomic variation within the Danish population.

`dbSNP.GoESP`: Allele frequency from the NHLBI Exome Sequencing Project (ESP) dataset. The NHLBI Grand Opportunity Exome Sequencing Project (GO-ESP) dataset contains 6,503 samples drawn from multiple ESP cohorts and represents all of the ESP exome variant data.

`dbSNP.GnomAD_exomes`: Allele frequency from the Genome Aggregation Database (gnomAD) exome dataset. The gnomAD exome dataset comprises a total of 16 million SNVs and 1.2 million indels from 125,748 exomes in 14 populations

`dbSNP.Siberian`: Allele frequency from a dataset that contains paired-end whole-genome sequencing data of 28 modern-day humans from Siberia and Western Russia.

`dbSNP.PRJEB37584`: Allele frequencies from the [PRJEB37584](https://www.ebi.ac.uk/ena/browser/view/PRJEB37584) dataset. The dataset contains genome-wide genotype analysis that identified copy number variations in cranial meningiomas in Chinese patients, and demonstrated diverse CNV burdens among individuals with diverse clinical features.

`dbSNP.SGDP_PRJ`: Allele frequencies from the [SGDP_PRJ](https://www.ebi.ac.uk/ena/browser/view/PRJEB9586) dataset. The Simons Genome Diversity Project dataset contains 263 C-panel fully public samples and 16 B-panel fully public samples for a total of 279 samples.

`dbSNP.1000Genomes`: Allele frequency from the 1000 Genomes Project dataset. The 1000 Genomes Project dataset contains 2,504 individuals from 26 populations across Africa, East Asia, Europe, and the Americas.

`dbSNP.dbGaP_PopFreq`: Allele frequency from the new source of dbGaP aggregated frequency data (>1 Million Subjects) provided by dbSNP.

`dbSNP.NorthernSweden`: Allele frequency from a dataset that contains 300 whole-genome sequenced human samples from the county of Vasterbotten in northern Sweden.

`dbSNP.HapMap`: Allele frequency from the HapMap project dataset. The International HapMap Project contains samples from African, Asian, or European populations.

`dbSNP.TOPMED`: Allele frequencies from the TOPMED dataset, which contains freeze 8 panel that includes about 158,000 individuals. The approximate ethnic breakdown is European(41%), African (31%), Hispanic or Latino (15%), East Asian (9%), and unknown (4%) ancestry.

`dbSNP.ALSPAC`: Allele frequency from the Avon Longitudinal Study of Parents and Children (ALSPAC) dataset. The UK10K - Avon Longitudinal Study of Parents and Children project contains 1927 sample including individuals obtained from the ALSPAC population. This population contains more than 14,000 mothers enrolled during pregnancy in 1991 and 1992.

`dbSNP.Qatari`: Allele frequency from the Qatar Genome dataset. The dataset contains initial mappings of the genomes of more than 1,000 Qatari nationals.

`dbSNP.MGP`: MGP contains aggregated information on 267 healthy individuals, representative of the Spanish population that were used as controls in the MGP (Medical Genome Project).

<br/>

### [cadd](http://cadd.gs.washington.edu)

A score >=0 that indicates deleteriousness of a variant. Variants with cadd > 15 are more likely to be deleterious.
See http://cadd.gs.washington.edu.

<br/>

### [caddIndel](http://cadd.gs.washington.edu)

A score >=0 that indicates deleteriousness of a variant. Variants with cadd > 15 are more likely to be deleterious.
See http://cadd.gs.washington.edu.

caddIndel scores are only defined for indels and MNPs. For SNPs, use the `cadd` field.

- Note that because Bystro decomposes MNPs into "SNP" records, the `caddIndel` field will occasionally be populated for SNPs (which are in fact part of MNPs in the CADD Indel dataset).

`caddIndel.alt`: The Bystro VCF-preprocessor's ALT record for this CADD site. This should always match the row's `alt` field value

`caddIndel.id`: The CADD VCF `ID`

`caddIndel.PHRED`: The CADD PHRED score for the insertion or deletion

<br/>

### clinvarVcf

ClinVar annotations, sourced from the ClinVar VCF dataset

`clinvarVcf.id`: The ClinVar VCF `ID`

`clinvarVcf.alt`: The Bystro VCF-preprocessor's ALT record for this ClinVar site. This should always match the row's `alt` field value

`clinvarVcf.AF_ESP`: Allele frequencies from GO-ESP

`clinvarVcf.AF_EXAC`: Allele frequencies from ExAC

`clinvarVcf.AF_TGP`: Allele frequencies from TGP

`clinvarVcf.ALLELEID`: The ClinVar Allele ID

`clinvarVcf.CLNDN`: ClinVar's preferred disease name for the concept specified by disease identifiers in CLNDISDB

`clinvarVcf.CLNDNINCL`: For included Variant : ClinVar's preferred disease name for the concept specified by disease identifiers in CLNDISDB

`clinvarVcf.CLNHGVS`: Top-level (primary assembly, alt, or patch) HGVS expression

`clinvarVcf.CLNREVSTAT`: ClinVar review status of germline classification for the Variation ID

`clinvarVcf.CLNSIG`: Aggregate germline classification for this single variant;

`clinvarVcf.CLNSIGCONF`: Conflicting germline classification for this single variant

`clinvarVcf.CLNVCSO`: Sequence Ontology id for variant type

`clinvarVcf.DBVARID`: NSV accessions from dbVar for the variant

`clinvarVcf.ORIGIN`: Allele origin. One or more of the following values may be added: 0 - unknown; 1 - germline; 2 - somatic; 4 - inherited; 8 - paternal; 16 - maternal; 32 - de-novo; 64 - biparental; 128 - uniparental; 256 - not-tested; 512 - tested-inconclusive; 1073741824 - other

`clinvarVcf.RS`: dbSNP ID (i.e. rs number)

<br/>

### (hg38-only) [LoGoFunc](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10688473/)

A machine learning method for predicting pathogenic GOF, pathogenic LOF, and neutral genetic variants, trained on a broad range of gene-, protein-, and variant-level features describing diverse biological characteristics.

`logofunc.id`: The LoGoFunc VCF `ID`

`logofunc.alt`: The Bystro VCF-preprocessor's ALT record for this LoGoFunc site. This should always match the row's `alt` field value

`logofunc.prediction`: The LoGoFunc prediction

`logofunc.neutral`: The LoGoFunc neutral score

`logofunc.gof`: The LoGoFunc gain of function (GOF) score

`logofunc.lof`: The LoGoFunc loss of function (LOF) score

<br/>

### (hg38-only) [GeneBass](<https://www.cell.com/cell-genomics/pdf/S2666-979X(22)00110-0.pdf>)

GeneBass provides statistics on the impact of genetic variants from gene-based phenome-wide association study (PheWAS) analysis results. See the link for more information.

`genebass.id`: The GeneBass VCF `ID`

`genebass.alt`: The Bystro VCF-preprocessor's ALT record for this GeneBass site. This should always match the row's `alt` field value

`genebass.phenocode`: The GeneBass phenotype code

`genebass.description`: The GeneBass description
