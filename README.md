# Bystro
## Using Bystro
For most users, we recommend https://bystro.io . The web app gives full access to all of Bystro's capabilities, and provides a convenient search/filtering interface.

## Installing Bystro
Follow the instructions in [INSTALL.md](INSTALL.md)

## Annotation (Output) Field Descriptions
Please read [FIELDS.md](FIELDS.md)

## The Bystro configuration file

- The config file describes the state of both the database and the annotation. It's required for annotating or building
- It has several keys:
- *tracks*: What your database contains, and what you annotate against. Tracks have a name, which must be unique, and a type, which doesn't need to be unique, but must be one of:
  - *type*:
    + *sparse*: Any bed file, or any file that can be mapped to chrom, chromStart, and chromEnd columns.
      + We can transform almost any file to fit this format. 
    + *score*: Accepts any wigFix file. 
      + Used for phastCons, phyloP
    + *cadd*: Accepts any CADD file, or Bystro's custom "bed-like" CADD file, which has 2 header lines, and chrom, chromStart, chromEnd columns, followed by standard CADD fields
      * CADD format: http://cadd.gs.washington.edu
    + *gene*: A UCSC gene track field (ex: knownGene, refGene, sgdGene).
    + The "source files" for this are created by an `sql_statement`:
      + Ex: SELECT * FROM hg38.refGene LEFT JOIN hg38.kgXref ON hg38.kgXref.refseq
    = hg38.refGene.name

### Directories and Files
These describe where the Bystro database and any source files are located.

1. `files_dir` : The parent folder in which our database source directories (which are used in creating the databsae) are located. This is only necessary when building a database
  * This directory must contain one folder, with the same name as each track being built

Ex:
```yml
files_dir: /path/to/
# ...
tracks:
  - name: refSeq
    type: gene
    local_files:
      - chr1.fa.gz
      - chr2.fa.gz
      - chr3.fa.gz
    # more refSeq track configuration
    # We then expect that all refSeq source file will be in /path/to/refSeq,
    # in which we find chr1.fa.gz, chr2.fa.gz, and chr3.fa.gz
```
  * It is also possible
2. `database_dir` : Where the database is located. Required
```
database_dir: /path/to/database/
```

# Statistics configuration
The "statistics" key describes the bystro-stats module configuration.
