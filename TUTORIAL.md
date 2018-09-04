# Bystro Tutorial

[Bystro](https://bystro.io) is a web applicatiom that simplifies genetic
analyses and handles large (Tb) sized experiments. Features include:

1. Genomic annotation of supplied variants
2. Relevant statistics for sequencing experiments
3. Search engine
4. Variant filtering
5. Simple saving of annotations and filtered annotation
6. Fast annotation and search performance

## Open Source

The open source command line annotator that powers bystro is located at the [bystro](https://github.com/akotlar/bystro).

## ACM-BCB 2018 Presentation

[Our ACM-BCB 2018 tutorial slides may be viewed here](tutorial/Presentation1.pdf)

## Bystro Manuscript

Read more examples and data are available in the manuscript, [Kotlar et al, Genome Biology, 2018](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-018-1387-3).

The data for the manuscript is located at the [bystro-paper](https://github.com/akotlar/bystro-paper)
repository.

## Help Guide

Major features are explained in the Bystro help guide which is accessed in the
[Guide link](https://bystro.io/guide) on the top banner.

## Register

Click [sign up](https://bystro.io/signup).

## Public Data

Bystro enables searching and filtering on publicly annotated genomic data
without registration.

Public data is accessed using the [Public link](https://bystro.io/public) on the top banner.

- 1000 Genomes Phase 3 (2,504 samples, 84.9M variants)
- Gemini Tutorial Trio data (3 samples, 13K variants)
- _drosophila Melanogaster_ fly lines from Drosopila Genetics Reference Panel 2 (205 samples, 3M variants)

## Annotate

Bystro accepts either `vcf` or `snp` formatted data for a number of model
organisms and genomic builds. When uploading the variants for annotation you
must select the correct genomic build. Once the annotation is complete you can
download the annotation while waiting for the annotation indexing to complete.
Annotation indexing is needed for the search and filter capabilities.

1. Click [submit](https://bystro.io/submit)
2. Choose Genome Assembly
3. Upload

## Annotation data sources

These include:

1. RefSeq provides details about the genes, transcripts, codons, amino acids, etc.
   - Predicted codon change, amino acid substitution
   - Nearest gene
   - Nearest transcription start site
2. Clinvar
3. CADD, PhyloP, PhasCons
4. dbSNP, gnomAD
5. COSMIC
6. heterozygotes IDs, homozygote IDs, missing IDs

Notes:

- missing value is `!`
- multiple transcripts are delimited by `;`
- multiallelic sites are delimited by `|`

A complete, abeit somewhat outdated, description of the fields are
[here](https://bystro.io/help/fields) and include:

**General Features**

```text
chrom           => chromosome
pos             => chromosome position
type            => type of variant (e.g., SNP, multi-allelelic)
discordant      => does the supplied reference match the genome assembly reference
alt             => supplied alternative alleles
trTv            => transition => 1, transversion => 2, neither => 0
heterozygotes   => heterozygotes IDs with the alternate allele
heterozygosity  => heterozygote count divided by non-missing allele number
homozygotes     => homozygotes IDs with the alternate allele
homozygosity    => homozygosity count divided by non-missing allele number
missingGenos    => IDs with missing genotyping for the site
missingness     => missing allele count divided by total allele counts
ac              => non-missing allele count
an              => non-missing genotypes
sampleMaf       => minor allele frequency of the supplied IDs
vcfPos          => the original VCF position (before sanitizing position of indels)
id              => copied from the VCF ID field
ref             => reference allele
```

**RefSeq Features**

Note, `name` refers to the transcript(s) and `name2` refers to gene symbols.
All of these data are directly downloaded from the UCSC refSeq track, and we
follow their naming conventions.

```
refSeq.siteType
refSeq.exonicAlleleFunction
refSeq.refCodon
refSeq.altCodon
refSeq.refAminoAcid
refSeq.altAminoAcid
refSeq.codonPosition
refSeq.codonNumber
refSeq.strand
refSeq.name
refSeq.name2
refSeq.description
refSeq.kgID
refSeq.mRNA
refSeq.spID
refSeq.spDisplayID
refSeq.protAcc
refSeq.rfamAcc
refSeq.tRnaName
refSeq.ensemblID
refSeq.clinvar.alleleID
refSeq.clinvar.phenotypeList
refSeq.clinvar.clinicalSignificance
refSeq.clinvar.type
refSeq.clinvar.origin
refSeq.clinvar.numberSubmitters
refSeq.clinvar.reviewStatus
refSeq.clinvar.chromStart
refSeq.clinvar.chromEnd
refSeq.gene.name2
refSeq.gene.pLi
refSeq.gene.pRec
refSeq.gene.pNull
refSeq.gene.lofTool
refSeq.gene.lofFdr
refSeq.gene.pHi
refSeq.gene.nonTCGA.pRec
refSeq.gene.nonTCGA.pNull
refSeq.gene.nonTCGA.pLi
refSeq.gene.nonPsych.pRec
refSeq.gene.nonPsych.pNull
refSeq.gene.nonPsych.pLi
refSeq.gene.gdi
refSeq.gene.cnv.score
refSeq.gene.cnv.flag
refSeq.gene.pmid
refSeq.gene.rvis
nearest.refSeq.name2
nearest.refSeq.name
nearest.refSeq.dist
nearestTss.refSeq.name2
nearestTss.refSeq.name
nearestTss.refSeq.dist
```

**Scores**

```
phastCons
phyloP
cadd
```

**gnomAD Features**

```
gnomad.genomes.alt
gnomad.genomes.id
gnomad.genomes.af
gnomad.genomes.an
gnomad.genomes.an_afr
gnomad.genomes.an_amr
gnomad.genomes.an_asj
gnomad.genomes.an_eas
gnomad.genomes.an_fin
gnomad.genomes.an_nfe
gnomad.genomes.an_oth
gnomad.genomes.an_sas
gnomad.genomes.an_male
gnomad.genomes.an_female
gnomad.genomes.af_afr
gnomad.genomes.af_amr
gnomad.genomes.af_asj
gnomad.genomes.af_eas
gnomad.genomes.af_fin
gnomad.genomes.af_nfe
gnomad.genomes.af_oth
gnomad.genomes.af_sas
gnomad.genomes.af_male
gnomad.genomes.af_female
gnomad.exomes.alt
gnomad.exomes.id
gnomad.exomes.af
gnomad.exomes.an
gnomad.exomes.an_afr
gnomad.exomes.an_amr
gnomad.exomes.an_asj
gnomad.exomes.an_eas
gnomad.exomes.an_fin
gnomad.exomes.an_nfe
gnomad.exomes.an_oth
gnomad.exomes.an_sas
gnomad.exomes.an_male
gnomad.exomes.an_female
gnomad.exomes.af_afr
gnomad.exomes.af_amr
gnomad.exomes.af_asj
gnomad.exomes.af_eas
gnomad.exomes.af_fin
gnomad.exomes.af_nfe
gnomad.exomes.af_oth
gnomad.exomes.af_sas
gnomad.exomes.af_male
gnomad.exomes.af_female
```

**dbSnp Features**

```
dbSNP.name
dbSNP.strand
dbSNP.observed
dbSNP.class
dbSNP.func
dbSNP.alleles
dbSNP.alleleNs
dbSNP.alleleFreqs
```

**ClinVar Features**

```
clinvar.alleleID
clinvar.phenotypeList
clinvar.clinicalSignificance
clinvar.type
clinvar.origin
clinvar.numberSubmitters
clinvar.reviewStatus
clinvar.referenceAllele
clinvar.alternateAllele
```

## Query

### Basic queries

Type a feature field followed by a filter criteria. For example, to select variants
that have a Combined Annotation Dependent Depletion Score (CADD) and minor allele
frequency (MAF) try the following.

1. `cadd > 20`
2. press return
3. `maf < 0.01`
4. press return

Equivalently, the same results are given with a single statement
`cadd > 20 AND maf < 0.01` or `cadd >20 maf < 0.01`. Notice that there is an
implied `AND` between the two filters (`cadd >20` and `maf < 0.01`) in the last
statement.

A somewhat more complex query where we are trying to find variants that have
been previously labeled as pathogenic in one of the data sources (e.g., ClinVar)
might look like this.

```text
cadd > 20 maf < .001 pathogenic expert review missense
cadd > 20 maf < .001 pathogenic expert’s review non-synonymous
cadd > 20 maf < .001 pathogen expert-reviewed nonsynonymous
```

Most of the time, puncutation and capitalization are unimportant; however, the
safest thing is to type an exact term.

For instance, the following statements are equivalent.

```text
early onset breast cancer
early-onset breast cancer
Early onset breast cancers
```

### Quering specific fields

To explicity search for gene symbol name try `refseq.name2: PSEN1`. The same
idea is applicable to searching other fields, for instance MAF might be more
applicable to our study population if we select the population with the same
genetic backround. For instance, to search MAF in Non-Finish Europeans try
`gnomad.genomes.af.nfe < 0.01`. Of course, any population in gnomAD could be
used. To search MAF in the entire gnomad database, use
`gnomad.genomes.af < 0.01`.

There are often multple ways to identify a single variant.

```text
Pathogenic nonsense Ehlers-Danlos
pathogenic nonsense E.D.S
pathogenic stopgain eds
```

### Using synonyms

Bystro supports user supplied synonyms. For example, we may want to label
some IDs as "cases" and others as "controls" to identify case-only results.

1. Designate case/control status using `custom-synonyms`
2. Assign a **cases** label and provide IDs of cases.
3. Assign a **controls** label and provide IDs of controls.
4. Search `cases -controls` to get case-only results. _Note that you cannot use a space before the minus (i.e., `-`) symbol._

Other things to **note** include:

- When multiple filters are supplied on a line and implicit `AND` is used to join them.
- The exact query that was processed will be displated below the search bar.
- By default, `missingGenos` is searchable.
- To avoid searching of the `missingGenos` field, restrict the ID search to heterozygote and homozygote fields by setting the definition of **case** and **control** to only search heterozygote or homozygote fields. For instance, `(heterozygote: ID || homozygote: ID)` will only search for the `ID` in either the heterozygote or homozygote fields.
- A site with one or more missing genotypes is counted as missing.

## Save and Download Annotation

### Save Annotation

To save any search results, click the floppy disk icon, which save the
filtered annotation within the web application. Saved annotations may be
accessed in the [Results link](https://bystro.io/results) in the top
banner.

### Download Annoation

To downlaod the results at any point, click on the icon of three horizontal
bars and select download.

### File description

If we uploaded a file called `file1` the following files will be included in
the downloaded tarball, which will be named `file1.tar`.

```text
file1.trim.vep.vcf.gz.file-log.log => output of VCF preprocessor
file1.annotation.log.txt           => annotation info (e.g., database used, time of annotation)
file1.annotation.tsv.gz            => tab-delimited annotation results
file1.sample_list                  => ids in original VCF file
file1.statistics.json              => same data as file1.statist.tsv in JSON format
file1.statistics.qc.tsv            => all samples 3 SD above and below the mean for ratio
                                      of Tr to Tv, Silent to Replacement, theta, etc.
file1.statistics.tsv               => per sample Tr, Tv, silent, and replacement stats
```

## Convert bystro annotation to VCF and linkage formats

This is an example of how to use the `bystro_to_vcf` program and `plink` to
convert the annoation first to a VCF fiel then to a linkage file format,
which are common formats for genomic analysis.

### Data for examples

```bash
git clone https://github.com/akotlar/bystro-tutorial
cd bystro-tutorial
```

### Compile `bystro_to_vcf`

This program converts the bystro annoation data to a vcf file. Assuming you are
in the repository.

```bash
cd src
gcc -Wall -O3 -pthreads -lm -lz bystro_to_vcf.c -o ../bystro_to_vcf
```

### Download genome build

These files are needed for the conversion of the bystro annotation to vcf file.

**Hg19 genome build**

```bash
curl -O https://s3.amazonaws.com/bystro-source/hg19.sdx
curl -O https://s3.amazonaws.com/bystro-source/hg19.seq
```

**Hg38 genome build**

```bash
curl -O https://s3.amazonaws.com/bystro-source/hg38.sdx
curl -O https://s3.amazonaws.com/bystro-source/hg38.seq
```

**Conversion Examples**

```bash
# convert annotation to vcf
bystro_to_vcf hg38.sdx <(pigz -d –c ann.tsv.gz) \
 | pigz -c > annotation.vcf.gz

# convert plink to binary linkage format
plink --vcf annotation.vcf.gz --keep-allele-order \
--const-fid seq --hwe 1e-6 midp --make-bed --out annotation.plink
```

Note, the `fam` file needs to be updated with family structure, sex, and
affectation or outcome data, as needed.

If you wish to drop samples, use

```bash
tail –n +18 file_name.statistics.tsv | cut -f1  | head > ids_to_keep
cat ids_to_keep | awk -F$"\t" '{print $1"\t"$1}' > ids_to_keep.with_fake_fam
plink --vcf you_vcf_file --no-fid --keep ids_to_keep.with_fake_fam
```

## Run SKAT

For the purposes of this example, let's add randomly assigned sex, affectation,
and covariates. Assuming that you are in the `bystro-tutorial` directory and
you have generated the `annotation.plink` file from above.

```bash
bin/run_skat_example.sh
```

## Machine Learning Classifier

See [this repository](https://bitbucket.org/akotlar/bystroml/src/master/).
