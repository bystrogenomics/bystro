### Beta 10.2 (2/26/18):

1. Add flexible fileProcessors support. Add a "fileProcessors" object to completely define the input pre-processor (such as vcf)
   - see: https://github.com/akotlar/bystro/blob/b10/config/hg19.clean.yml
2. Support output of a <your_annotation>.sample_list file, which will contain the complete list of inputted samples (found in the .snp or .vcf input file). This will be eventually used to export to vcf/plink.
3. Backported various simplifications from master branch (b11).

### Beta 10.1 (2/26/18):

1. Added SAS population to gnomad.exomes (thanks David Okou)

2. hg38: refetched all sources as of 2/7/18

- 2/5/18 variant_summary.txt.gz Clinvar update
- gnomAD r2.0.2
- dbSNP150

3. hg19: refetched all sources as of 2/26/18

- 2/19/18 variant_summary.txt.gz Clinvar update
- gnomAD r2.0.2
- dbSNP150

### Beta 9 (9/13/17):

1. Added heterozygosity, homozygosity, missingness, sampleMaf fields
2. Added gnomad.genomes and gnomad.exomes for hg38 (lifted over) and hg19

Breaking Changes:

1. bystro-vcf and bystro-snp output a header, to simplify using these programs' outputs
2. The additiona fields (heterozygosity, homozygosity, missingness, sampleMaf) affect bystro-vcf and bystro-snp output order
3. bystro-vcf now has a maximum limit of 10 alleles (9 non-reference) for any multiallelic

- This allows for substantial speedup of the slowest function (the makeHetHomozygotes function, which calculates heterozygosity, homozygosity)

Note: As a result of the use of floats, and the current limitation on Perl's best msgpack implementation (no float32 serialization), db size has grown ~20GB due to gnomad. Unfortunately, since floats don't compress well, and I wanted to avoid modifying the source data by rounding, the compressed size .tar.gz has grown by a similar order (~10-20GB).

Performance Improvements:
~ 15% reduction in CPU usage for bystro-vcf for vcf files with many samples (15% on 1000 Genomes, 2504 samples), and a slight improvement in throughput.

### Beta 8 (7/12/17):

Breaking Changes:

1. Switchecd .snp annotation path to bystro-snp package. Multiallelics are decomposed into bi-allic sites (where one of the alleles is the reference)

- Heterozygosity/homozygosity for such decomposition is handled relative to the minor allele
- Ex: A sample with 2 different non-reference alleles at that position will be given a het call for the first allele, and a het call for the 2nd, and both sites will be labeled MULTIALLELIC

2. This means that unlike previously, annotations from .snp files are not guaranteed to have 1 row per position (VCF annotations were always this way)
3. The "MULTIALLELIC" label now added to bystro-vcf: any sites with 2 or more non-reference alleles will be labelled as such, matching the bystro-snp path.
4. A new column, "trTv" added. This contains a 0 if the site was neither a transition or transversion, 1 if transition, 2 if transversion.
5. The bystro-statistics calculator now uses the "trTv" field (if available) to calculate tr/tv ratio. Results are always the same, except now MULTIALLELIC sites are not considered.

### Beta 7 (6/22/17):

Breaking Changes:

1. Renamed "missing" to "missingGenos". "missing:fieldName" may be used for excluding fields during search.

### Beta 6 (6/18/17):

Breaking Changes:

1. 7th column, "missing" added. This contains any samples whose genotypes are missing ('.|.', '././', or 'N')

- In multiallelic cases, any missing samples will be reported for each allele

2. When annotating snp files, Bystro not longer drops samples with <.95 confidence. This made denovo (or case -control) queries innacurate

Improvements:

1. bystro-vcf uses a much more restrictive approach to calling indels

- Single-variant sites can only have 1 base padding in case of insertions, and in case of deletions the alt must be 1 in length
- In case of multiallelics where the reference is longer than 1bp (as happens when one of the variants is a deletion), the insertion must begin after the 1st base (padding)
- In cases of multiallelics where the reference is longer than 1bp and the alt is longer than 1bp (but shorter than reference), the deletion must similarly occur beginning with the 2nd bse
- If the reference is longer than 1bp, if the site is an insertion it will be skipped
- If the alt is longer than 1bp and smaller than the reference (deletion, but with extra padding), it will be skipped

Bug fixes:

1. By the above, in highly repetitive regions, bystro-vcf will no longer call the allele to the right of the common sequence shared between the ref and the allele

- By VCF spec it is not clear that this is strictly a bug; in fact VCF 4.2 documentation opens with a multiallelic site (AC / ACT) that is called in this way, making it impossible to conclusively call alleles with repetitive sequences
- However, looking at the gnomAD team's interpretation of multiallelic sites, the current approach seems more consistent.
- I also believe that it is appropriate to indicate to people that they should never place an allele with more than 1 base padding, since this decreases the value of having a pos column, and is in general a ridiculous way to represent variants.

Performance:
1/2 CPU use while parsing vcf files containing many samples
Somewhat faster snp parsing of the same

Notes: bystro-vcf now supports custom FILTER'ing (defaults to PASS and "." as being acceptable), based on either exclusion criteria (excludeFilter) and inclusion criteria (keepFilter), ala vcftools/bcftools. This will be exposed in the web interface at some point. Similarly bystro-vcf now supports keeping id and info columns. See bystro-vcf commit history for more detail (we'll be writing documetnation for bystro-vcf in the future, time permitting).

### Beta 5:

Breaking Changes:

#### VCF handling

bystro-vcf now allows only PASS or "." FILTER column values

### Beta 4:

Breaking Changes:

#### Search mapping

refSeq.name2, refSeq.nearest.name2, and refSeq.spDisplayID are no longer indexed using edge n-grams. This means that by default, it is searched exactly (except capitalization doesn't matter).

To search refSeq.name2 (or any field) by prefix, simply type \* (star) after the term (ex: dlg\*)

This was done because many users reported getting unexpected results, and to standardize quotes as being used for phrases, rather than exact matches.

### Beta 3:

Breaking Changes:

#### Clinvar

Normalized clinvar header names. This uses the new fieldMap property, which can be applied to any header column values found in any sparse track (.bed-like file), and which replaces the required_fields_map property, which worked only for chromStart, chromEnd, and chrom.

Sparse tracks still require chromStart, chromEnd, and chrom fields, either to be present, or to be provided via a fieldMap transformation.

```yaml
fieldMap:
  "#AlleleID": alleleID
  AlternateAllele: alternateAllele
  Chromosome: chrom
  ClinicalSignificance: clinicalSignificance
  Origin: origin
  OtherIDs: otherIDs
  PhenotypeIDS: phenotypeIDs
  NumberSubmitters: numberSubmitters
  PhenotypeList: phenotypeList
  ReferenceAllele: referenceAllele
  ReviewStatus: reviewStatus
  Start: chromStart
  Stop: chromEnd
  Type: type
```

Update clinvar track features

```yaml
features:
  - alleleID: number
  - phenotypeList
  - clinicalSignificance
  - type
  - origin
  - numberSubmitters: number
  - reviewStatus
  - referenceAllele
  - alternateAllele
```

#### RefSeq

Replaced geneSymbol with name2. name2 is provided by UCSC in the refGene database, and is the equivalent of the kgXref geneSymbol value, except is sometimes more complete. Since we are LEFT joining on refGene (aka providing the RefSeq transcripts), it makes sense to use the RefSeq/refGene value whenever possible

#### RefSeq.nearest

Replaced geneSymbol with name2. See RefSeq note above.

#### RefSeq.clinvar

Updated refSeq.clinvar overlap records:

```yaml
join:
  features:
    - alleleID
    - phenotypeList
    - clinicalSignificance
    - type
    - origin
    - numberSubmitters
    - reviewStatus
    - chromStart
    - chromEnd
```
