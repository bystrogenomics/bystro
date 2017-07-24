# Bystro
## Annotator Output
#### Some Bystro output columns are found regardless of the included tracks

1. chrom
    - The chromosome name (chr1, chr2, etc)

2. pos
    - The position. For deletions this is the first base deleted, for insertions this is the base just before the insertion.
    - This is different from VCF spec, and much less complicated

3. type
    - The variant call.
    - Possibilities: SNP, INS, DEL, MULTIALLELIC, DENOVO_SNP, DENOVO_INS, etc.

4. discordant
    - Whether the assembly matches the reference alelle given by the user's input file.

5. alt
    - The alternate allele(s). This is taken directly from the input file.heterozygotes

5. homozygotes
    - If the sample has the given allele single-copy.
    - In multiallelic case, phased on allele

6. heterozygotes
    - If the sample has the given allele in copy nunmber matching the # of chromosomes
    - In multiallelic case, phased on allele

### The Bystro configuration file

- The config file describes the state of both the database and the annotation. It's required for annotating or buildign
- It has several keys:
- *tracks*: What your database contains, and what you annotate against. Tracks have a name, which must be unique, and a type, which doesn't need to be unique
  - *type*: A track needs to have a type
    + *sparse*: Accepts any bed file, or any file that has at least a valid chrom, chromStart, and chromEnd. We can transform almost any file to fit this format, TODO: give example below.
    + *score*: Accepts any wigFix file. 
      + Used for phastCons, phyloP
    + *cadd*: Accepts any CADD file, or Bystro's custom "bed-like" CADD file (TODO: DESCRIBE)
      * CADD format: http://cadd.gs.washington.edu
    + *gene*: A UCSC gene track, either knownGene, or refGene. The "source files" for this is an `sql_statement` key assigned to this track (described below)

# Directories and Files
These describe where the Bystro database and any source files are located.

1. `files_dir` : Where our source files are located. This is only necessary when building a new database

Ex:
```
files_dir: /path/to/files/
```

2. `database_dir` : Where the database is located. Required
```
database_dir: /path/to/database/
```