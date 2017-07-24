Tracks
---
---
## Track Types

We expose 3 kinds of general purpose tracks
### General Tracks 

1. Sparse (sparse)
  - These are any tracks that have unique info per base
  - These must be .bed - like format, containing the following fields (no other fields are required)
    + chrom
    + chromStart
    + chromEnd
  - Ex: snp146

2. Score (score)
 - Any fixed wiggle format track
   + ex: PhyloP, PhastCons

We also have several "private" track types. These are still defined in the config file, but are just our special implementations of the above 3.

### Special Tracks
These are special cases of the above tracks

1. Reference: the genome assembly (*Only 1 per configuration file*)
  - Accepts: multi-fasta file

2. Gene:
  - Accepts: UCSC gene track, such as refGene
  - Stores: any features defined in the track configuration, that are present in the source file\

3. CADD 
  - Accepts:
    - CADD format file (1-based) (http://krishna.gs.washington.edu/download/CADD/v1.3/whole_genome_SNVs.tsv.gz)
    - Bed-like format, where first 3 header columns (after version line) are chrom, chromStart, chromEnd (0-based, half-open format)

# Building tracks
#### Tracks are stored in a YAML configuration file, such as the file below
```yaml
---
assembly: hg19
build_author: ec2-user
build_date: 2017-02-08T03:01:00
chromosomes:
- chr1
- chr2
- chr3
- chr4
- chr5
- chr6
- chr7
- chr8
- chr9
- chr10
- chr11
- chr12
- chr13
- chr14
- chr15
- chr16
- chr17
- chr18
- chr19
- chr20
- chr21
- chr22
- chrM
- chrX
- chrY
database_dir: /path/to/somewhere/
files_dir: /path/to/somewhere/
statistics:
  dbSNPnameField: dbSNP.name
  exonicAlleleFunctionField: refSeq.exonicAlleleFunction
  outputExtensions:
    json: .statistics.json
    qc: .statistics.qc.tab
    tab: .statistics.tab
  refTrackField: ref
  siteTypeField: refSeq.siteType
temp_dir: ~ #Optional
tracks:
- build_author: ec2-user
  build_date: 2017-02-08T03:01:00
  fetch_date: 2017-02-04T22:36:00
  local_files:
  - chr*.fa.gz
  name: ref
  remote_dir: http://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/
  remote_files:
  - chr1.fa.gz
  - chr2.fa.gz
  - chr3.fa.gz
  - chr4.fa.gz
  - chr5.fa.gz
  - chr6.fa.gz
  - chr7.fa.gz
  - chr8.fa.gz
  - chr9.fa.gz
  - chr10.fa.gz
  - chr11.fa.gz
  - chr12.fa.gz
  - chr13.fa.gz
  - chr14.fa.gz
  - chr15.fa.gz
  - chr16.fa.gz
  - chr17.fa.gz
  - chr18.fa.gz
  - chr19.fa.gz
  - chr20.fa.gz
  - chr21.fa.gz
  - chr22.fa.gz
  - chrM.fa.gz
  - chrX.fa.gz
  - chrY.fa.gz
  type: reference
  version: 2
- build_author: ec2-user
  build_date: 2017-02-08T03:01:00
  features:
  - kgID
  - mRNA
  - spID
  - spDisplayID
  - geneSymbol
  - refseq
  - protAcc
  - description
  - rfamAcc
  - name
  fetch_date: 2017-02-04T17:06:00
  join:
    features:
    - PhenotypeIDS
    - OtherIDs
    track: clinvar
  local_files:
  - hg19.refGene.chr*.gz
  name: refSeq
  nearest:
  - name
  - geneSymbol
  sql_statement: SELECT * FROM hg19.refGene LEFT JOIN hg19.kgXref ON hg19.kgXref.refseq
    = hg19.refGene.name
  type: gene
  version: 2
- build_author: ec2-user
  build_date: 2017-02-08T03:01:00
  fetch_date: 2017-02-04T16:52:00
  local_files:
  - chr*.phastCons100way.wigFix.gz
  name: phastCons
  remote_dir: http://hgdownload.soe.ucsc.edu/goldenPath/hg19/phastCons100way/hg19.100way.phastCons/
  remote_files:
  - chr1.phastCons100way.wigFix.gz
  - chr2.phastCons100way.wigFix.gz
  - chr3.phastCons100way.wigFix.gz
  - chr4.phastCons100way.wigFix.gz
  - chr5.phastCons100way.wigFix.gz
  - chr6.phastCons100way.wigFix.gz
  - chr7.phastCons100way.wigFix.gz
  - chr8.phastCons100way.wigFix.gz
  - chr9.phastCons100way.wigFix.gz
  - chr10.phastCons100way.wigFix.gz
  - chr11.phastCons100way.wigFix.gz
  - chr12.phastCons100way.wigFix.gz
  - chr13.phastCons100way.wigFix.gz
  - chr14.phastCons100way.wigFix.gz
  - chr15.phastCons100way.wigFix.gz
  - chr16.phastCons100way.wigFix.gz
  - chr17.phastCons100way.wigFix.gz
  - chr18.phastCons100way.wigFix.gz
  - chr19.phastCons100way.wigFix.gz
  - chr20.phastCons100way.wigFix.gz
  - chr21.phastCons100way.wigFix.gz
  - chr22.phastCons100way.wigFix.gz
  - chrX.phastCons100way.wigFix.gz
  - chrY.phastCons100way.wigFix.gz
  - chrM.phastCons100way.wigFix.gz
  type: score
  version: 2
- build_author: ec2-user
  build_date: 2017-02-08T03:01:00
  fetch_date: 2017-02-03T20:57:00
  local_files:
  - chr*.phyloP100way.wigFix.gz
  name: phyloP
  remote_dir: http://hgdownload.soe.ucsc.edu/goldenPath/hg19/phyloP100way/hg19.100way.phyloP100way/
  remote_files:
  - chr1.phyloP100way.wigFix.gz
  - chr2.phyloP100way.wigFix.gz
  - chr3.phyloP100way.wigFix.gz
  - chr4.phyloP100way.wigFix.gz
  - chr5.phyloP100way.wigFix.gz
  - chr6.phyloP100way.wigFix.gz
  - chr7.phyloP100way.wigFix.gz
  - chr8.phyloP100way.wigFix.gz
  - chr9.phyloP100way.wigFix.gz
  - chr10.phyloP100way.wigFix.gz
  - chr11.phyloP100way.wigFix.gz
  - chr12.phyloP100way.wigFix.gz
  - chr13.phyloP100way.wigFix.gz
  - chr14.phyloP100way.wigFix.gz
  - chr15.phyloP100way.wigFix.gz
  - chr16.phyloP100way.wigFix.gz
  - chr17.phyloP100way.wigFix.gz
  - chr18.phyloP100way.wigFix.gz
  - chr19.phyloP100way.wigFix.gz
  - chr20.phyloP100way.wigFix.gz
  - chr21.phyloP100way.wigFix.gz
  - chr22.phyloP100way.wigFix.gz
  - chrX.phyloP100way.wigFix.gz
  - chrY.phyloP100way.wigFix.gz
  - chrM.phyloP100way.wigFix.gz
  type: score
  version: 2
- build_author: ec2-user
  build_date: 2017-02-08T03:01:00
  local_files:
  - whole_genome_SNVs.tsv.bed.chr*.organized-by-chr.txt.sorted.txt.gz
  name: cadd
  sort_date: 2017-01-20T16:06:00
  sorted_guaranteed: 1
  type: cadd
  version: 2
- build_author: ec2-user
  build_date: 2017-02-08T03:01:00
  build_field_transformations:
    alleleFreqs: split [,]
    alleleNs: split [,]
    alleles: split [,]
    func: split [,]
    observed: split [\/]
  features:
  - name
  - strand
  - observed
  - class
  - func
  - alleles
  - alleleNs: number
  - alleleFreqs: number
  fetch_date: 2017-02-04T21:35:00
  local_files:
  - hg19.snp147.chr*.gz
  name: dbSNP
  sql_statement: SELECT * FROM hg19.snp147
  type: sparse
  version: 2
- based: 1
  build_author: ec2-user
  build_date: 2017-02-08T03:01:00
  build_field_transformations:
    Chromosome: chr .
    OtherIDs: split [;,]
    PhenotypeIDS: split [;,]
  build_row_filters:
    Assembly: == GRCh37
    Chromosome: != MT
  build_field_filters:
    Assembly: == GRCh37
    Chromosome: != MT
  fieldMap:
    '#AlleleID': alleleID
    AlternateAllele: alternateAllele
    Chromosome: chrom
    ClinicalSignificance: clinicalSignificance
    Origin: origin
    OtherIDs: otherIDs
    PhenotypeIDS: phenotypeIDs
    PhenotypeList: phenotypeList
    ReferenceAllele: referenceAllele
    ReviewStatus: reviewStatus
    Start: chromStart
    Stop: chromEnd
    Type: type
  features:
  - alleleID: number
  - phenotypeList
  - clinicalSignificance
  - type
  - origin
  - reviewStatus
  - otherIDs
  - phenotypeIDs
  - referenceAllele
  - alternateAllele
  fetch_date: 2017-02-04T16:51:00
  local_files:
  - variant_summary.txt.gz
  name: clinvar
  remote_files:
  - ftp://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
  required_fields_map:
    chrom: Chromosome
    chromEnd: Stop
    chromStart: Start
  type: sparse
  version: 2
version: 2
```

A genome build can be run by executing the ./bin/build_genome_assembly.pl program using the configuration file

# YAML properties
## features
#### A list of field nams, either the original found in the source file, or the renamed value given in fieldMap

##### Ex1: original field name
```yaml
  features:
  - ClinicalSignificance #Must exist in the source file
```

##### Ex2: renamed field
```yaml
fieldMap:
    '#AlleleID': alleleID
    AlternateAllele: alternateAllele
    Chromosome: chrom
    ClinicalSignificance: clinicalSignificance
    Origin: origin
    OtherIDs: otherIDs
    PhenotypeIDS: phenotypeIDs
    PhenotypeList: phenotypeList
    ReferenceAllele: referenceAllele
    ReviewStatus: reviewStatus
    Start: chromStart
    Stop: chromEnd
    Type: type
  features:
  # If fieldMap is defined, the features specified here should be the renamed values
  # Since 'AlleleID' was renamed alleledID, use that name
  - alleleID: number #Note that a field data type can be specified after the colon
```
# General YAML build configuration properties
## build_row_filters
#### Simple boolean opeartions that determine whether or not a row will be included in the database
##### Warning: Does not accept renamed fields, unlike features

##### Ex: 
```yaml
# Don't include any rows in the source file, whose Assembly equals GRCh37, or Chromosome equals MT
build_row_filters:
    Assembly: == GRCh37
    Chromosome: != MT
```
## build_field_transformations
#### Modify the value of any field
##### Current operations:
  * split
    * split the field, on any regular expression, in the form of "split rePattern"
  * "."
    * concatenate the value of any field to some string, in the form of ". somestring" or "somestring ." for prepend/append resp.

#### Ex:
```yaml
build_field_transformations:
    # If fieldMap is used for this track, these field names should be the renamed field names
    chrom: chr .
    clinicalSignificance: split [;]
    otherIDs: split [;,]
    phenotypeIDs: split [;,]
    phenotypeList: split [;]
```

# Gene track-specific YAML configuration properties
## join
#### Allows you to add any track to the gene track
##### (currently joined track must define features, so sparse or gene)

##### Ex:
```yaml
join:
    # These should match features defined in the clinvar track "features" property
    features:
    - phenotypeIDs
    - otherIDs
    - alleleID
    track: clinvar
```

## nearest
#### If no transcript exists at a given position store the nearest transcript
##### When transcripts are equidistant, the downstream one is chosen
##### If the nearest transcripts are overlapping (multiple transcripts at one location), all of them will be stored

##### Ex:
```yaml
nearest:
  - name
  - geneSymbol
```
