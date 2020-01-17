# Bystro [![DOI](https://zenodo.org/badge/98203430.svg)](https://zenodo.org/badge/latestdoi/98203430) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/0859a24d422a4d80a2ad6721e294aa35)](https://app.codacy.com/app/akotlar/bystro?utm_source=github.com&utm_medium=referral&utm_content=akotlar/bystro&utm_campaign=badger)

TLDR; 1,000x+ faster than VEP, more complete annotation + online search (https://bystro.io) for datasets of up to dozens of terabytes.

![Bystro Performance](https://i.imgur.com/ve8hUF8.png)
## Bystro Publication

For datasets and scripts used, please visit [github.com/bystro-paper](https://github.com/akotlar/bystro-paper)

If using Bystro, please cite [Kotlar et al, Genome Biology, 2018](https://doi.org/10.1186/s13059-018-1387-3)

## Web Tutorial

Start here: [TUTORIAL.md](TUTORIAL.md)

For most users, we recommend https://bystro.io .

The web app gives full access to all of Bystro's capabilities, provides a convenient search/filtering interface, supports large data sets (tested up to 890GB uncompressed/129GB compressed), and has excellent performance.

## Installing Bystro

### Current release branch
The most up-to-date release branch is [found here](https://github.com/akotlar/bystro/tree/b10). You may also download a [release archive](https://github.com/akotlar/bystro/releases) and install from there.

### Master branch instruction
This branch is under active development, but should be stable enough to build and annotate from.

Please read: [INSTALL.md](INSTALL.md)

Bystro relies on pluggable (via Bystro's YAML config) pre-processors to normalize variant inputs (**dealing with VCF issues such as padding**), calculate whether a site is a transition or transversion, calculate sample maf, identify hets/homozygotes/missing samples, calculate heterozygosity, homozygosity, missingness, and more.

1. VCF format: [Bystro-Vcf](https://github.com/akotlar/bystro-vcf)
2. SNP format: [Bystro-SNP](https://github.com/akotlar/bystro-snp)
3. Create your own to support other formats!

## Annotation (Output) Field Descriptions

Please read [FIELDS.md](FIELDS.md)

## The Bystro configuration file

- The config file describes the state of both the database and the annotation. It's required for annotating or building
- It has several keys:

  - `tracks`: The highest level organization for database values. Tracks have a `name` property, which must be unique, and a `type`, which must be one of:
    - _sparse_: Any bed file, or any file that can be mapped to chrom, chromStart, and chromEnd columns.
      - This is used for dbSNP, and Clinvar records, but many files can be fit this format.
      - Mapping fields can be managed by the `fieldMap` key
    - _score_: Accepts any wigFix file.
      - Used for phastCons, phyloP
    - _cadd_:
      - Accepts any CADD file, or Bystro's custom "bed-like" CADD file, which has 2 header lines, and chrom, chromStart, chromEnd columns, followed by standard CADD fields
      - CADD format: http://cadd.gs.washington.edu
    - _gene_: A UCSC gene track field (ex: knownGene, refGene, sgdGene).
      - The `local_files` for this are created using an `sql_statement`
      - Ex: `SELECT * FROM hg38.refGene LEFT JOIN hg38.kgXref ON hg38.kgXref.refseq = hg38.refGene.name`
  - `chromosomes`: The allowable chromosomes.

    - Each row of every track must be identified by these chromosomes (during building)
    - Each row of any input file submitted for annotation must also be "" "" (during annotation)
    - However, Bystro is flexible about the **chr** prefix

    **Ex:** For the following config

    ```yaml
    chromosomes:
      - chr1
      - chr2
      - chr3
    ```

    Only chr1, chr2, and chr3 will be accepted. However, Bystro tries to make your life easy

    1. We currently follow UCSC conventions for `chromosomes`, meaning they should be prepended by **chr**
    2. Bystro will automatically append **chr** to chromosomes read from an input file during annotation.
    3. Bystro allows the transformation of any field during building, configurable in the YAML config file for that assembly, making it easy to prepend **chr** to the source file chromosome field

    Ex: Clinvar doesn't have a **chr** prefix, so during building we specify:

    ```yaml
    tracks:
      - name: clinvar
        build_field_transformations:
          chrom: chr .
        fieldMap:
          Chromosome: chrom
    ```

    Here `fieldMap` allows us to rename header fields, and `build_field_transformations` allows us to define a prepend operation (`chr .` can be interpreted as the perl command `"chr" . $chrom`)

    So: input files do **not** need to have their chromosomes prepended by **chr**. Bystro will normalize the name.

    In this example chromosomes `1` and `chr1` will be built/annotated, but `1_rand` will not.

### Directories and Files

These describe where the Bystro database and any source files are located.

1. `files_dir` : The parent folder within which each track's `local_files` are located

- Bystro automatically checks for `local_files` at `parent/trackName/file`

  **Ex:** For the config file containing

  ```yaml
  files_dir: /path/to/files/
  track:
    - name: refSeq
      local_files:
        - hg19.refGene.chr1.gz
        # and more files
  ```

  Bystro will expect files in `/path/to/files/refSeq/hg19.refGene.chr1.gz`

2. `database_dir` : Each database is held within `database_dir`, in a folder of the name `assembly`

   **Ex:** For the config file containing

   ```yaml
   assembly: hg19
   database_dir: /path/to/databases/
   ```

   Bystro will look for the database `/path/to/databases/hg19`
