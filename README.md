# Bystro [![DOI](https://zenodo.org/badge/98203430.svg)](https://zenodo.org/badge/latestdoi/98203430) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/0859a24d422a4d80a2ad6721e294aa35)](https://app.codacy.com/app/akotlar/bystro?utm_source=github.com&utm_medium=referral&utm_content=akotlar/bystro&utm_campaign=badger)

TLDR; 1,000x+ faster than VEP, more complete annotation + online search (https://bystro.io) for datasets of up to 47TB (compressed) online, or petabytes offline.

![Bystro Performance](https://i.imgur.com/ve8hUF8.png)

## Bystro Publication

For datasets and scripts used, please visit [github.com/bystro-paper](https://github.com/akotlar/bystro-paper)

If using Bystro, please cite [Kotlar et al, Genome Biology, 2018](https://doi.org/10.1186/s13059-018-1387-3)

## Web Tutorial

Start here: [TUTORIAL.md](TUTORIAL.md)

For most users, we recommend https://bystro.io .

The web app gives full access to all of Bystro's capabilities, provides a convenient search/filtering interface, supports large data sets (tested up to 890GB uncompressed/129GB compressed), and has excellent performance.

## Installing Bystro

Bystro consists of 2 main components: the Bystro Python package, which consists of the Bystro ML library, CLI tool, and a collection of easy to use biology tools including global ancestry and the Bystro annotator (Perl).

The Bystro Python package also gives the ability to launch workers to process jobs from the Bystro API server, but this is not necessary for most users.

### Installing the Bystro Python libraries and CLI tools

To install the Bystro Python package, run:

```sh
pip install --pre bystro
```

The Bystro ancestry CLI `score` tool (`bystro-api ancestry score`) parses VCF files to generate dosage matrices. This requires `bystro-vcf`, a Go program which can be installed with:

```sh
# Requires Go: install from https://golang.org/doc/install
go install github.com/bystrogenomics/bystro-vcf@2.2.2
```

Bystro is compatible with Linux and MacOS. Windows support is experimental. If you are installing on MacOS as a native binary (Arm), you will need to install the following additional dependencies:

```sh
brew install cmake
```

Please refer to [INSTALL.md](INSTALL.md) for more details.

### Installing the Bystro Annotator

Please refer to [INSTALL.md](INSTALL.md) for instructions on how to install the Bystro annotator.

### File support

Bystro relies on pluggable (via Bystro's YAML config) pre-processors to normalize variant inputs (**dealing with VCF issues such as padding**), calculate whether a site is a transition or transversion, calculate sample maf, identify hets/homozygotes/missing samples, calculate heterozygosity, homozygosity, missingness, and more.

1. VCF format: [Bystro-Vcf](https://github.com/bystrogenomics/bystro-vcf/tree/2.2.2)
2. SNP format: [Bystro-SNP](https://github.com/akotlar/bystro-snp)
3. Create your own to support other formats!

## Annotation (Output) Field Descriptions

Please read [FIELDS.md](FIELDS.md)

## The Bystro configuration file

- The config file describes the state of both the database and the annotation. It's required for annotating or building
- It has several keys:

  - `tracks`: The highest level organization for database values. Tracks have a `name` property, which must be unique, and a `type`, which must be one of:

    - _sparse_: A bed file, or any file that can be mapped to `chrom`, `chromStart`, and `chromEnd` columns.
      - This is used for dbSNP, and Clinvar records, but many files can be fit this format.
      - Mapping fields can be managed by the `fieldMap` key
    - _score_: A wigFix file.
      - Used for phastCons, phyloP
    - _cadd_:
      - A CADD file, or Bystro's custom "bed-like" CADD file, which has 2 header lines, and chrom, chromStart, chromEnd columns, followed by standard CADD fields
      - CADD format: http://cadd.gs.washington.edu
    - _gene_: A UCSC gene track table (ex: knownGene, refGene, sgdGene) stored as a tab separated output, with column names as columns. Conversion from SQL to the expected tab-delimited format is controlled by bin/bystro-utils.pl, which will automatically fetch the requested sql, and generate the tab-delimited output.

      For instance: For a config file that has the following track

      ```
      chromosomes:
        - chr1
      tracks:
        tracks:
        - name: refSeq
          type: gene
          utils:
          - args:
              connection:
                database: hg19
              sql: SELECT r.*, (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.kgID, '')) SEPARATOR
                ';') FROM kgXref x WHERE x.refseq=r.name) AS kgID, (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.description,
                '')) SEPARATOR ';') FROM kgXref x WHERE x.refseq=r.name) AS description,
                (SELECT GROUP_CONCAT(DISTINCT(NULLIF(e.value, '')) SEPARATOR ';') FROM knownToEnsembl
                e JOIN kgXref x ON x.kgID = e.name WHERE x.refseq = r.name) AS ensemblID,
                (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.tRnaName, '')) SEPARATOR ';') FROM
                kgXref x WHERE x.refseq=r.name) AS tRnaName, (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.spID,
                '')) SEPARATOR ';') FROM kgXref x WHERE x.refseq=r.name) AS spID, (SELECT
                GROUP_CONCAT(DISTINCT(NULLIF(x.spDisplayID, '')) SEPARATOR ';') FROM kgXref
                x WHERE x.refseq=r.name) AS spDisplayID, (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.protAcc,
                '')) SEPARATOR ';') FROM kgXref x WHERE x.refseq=r.name) AS protAcc, (SELECT
                GROUP_CONCAT(DISTINCT(NULLIF(x.mRNA, '')) SEPARATOR ';') FROM kgXref x WHERE
                x.refseq=r.name) AS mRNA, (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.rfamAcc,
                '')) SEPARATOR ';') FROM kgXref x WHERE x.refseq=r.name) AS rfamAcc FROM
                refGene r WHERE chrom=%chromosomes%;
      ```

      Running `bin/bystro-utils.pl --config <path/to/this/config> ` will result in the following config:

      ```
      chromosomes:
        - chr1
      tracks:
        tracks:
        - name: refSeq
          type: gene
          local_files:
            - hg19.kgXref.chr1.gz
            name: refSeq
            type: gene
            utils:
            - args:
                connection:
                  database: hg19
                sql: SELECT r.*, (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.kgID, '')) SEPARATOR
                  ';') FROM kgXref x WHERE x.refseq=r.name) AS kgID, (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.description,
                  '')) SEPARATOR ';') FROM kgXref x WHERE x.refseq=r.name) AS description,
                  (SELECT GROUP_CONCAT(DISTINCT(NULLIF(e.value, '')) SEPARATOR ';') FROM knownToEnsembl
                  e JOIN kgXref x ON x.kgID = e.name WHERE x.refseq = r.name) AS ensemblID,
                  (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.tRnaName, '')) SEPARATOR ';') FROM
                  kgXref x WHERE x.refseq=r.name) AS tRnaName, (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.spID,
                  '')) SEPARATOR ';') FROM kgXref x WHERE x.refseq=r.name) AS spID, (SELECT
                  GROUP_CONCAT(DISTINCT(NULLIF(x.spDisplayID, '')) SEPARATOR ';') FROM kgXref
                  x WHERE x.refseq=r.name) AS spDisplayID, (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.protAcc,
                  '')) SEPARATOR ';') FROM kgXref x WHERE x.refseq=r.name) AS protAcc, (SELECT
                  GROUP_CONCAT(DISTINCT(NULLIF(x.mRNA, '')) SEPARATOR ';') FROM kgXref x WHERE
                  x.refseq=r.name) AS mRNA, (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.rfamAcc,
                  '')) SEPARATOR ';') FROM kgXref x WHERE x.refseq=r.name) AS rfamAcc FROM
                  refGene r WHERE chrom=%chromosomes%;
              completed: <date fetched>
              name: fetch
      ```

      `hg19.kgXref.chr1.gz` will contain:

      ```sv
      bin	name	chrom	strand	txStart	txEnd	cdsStart	cdsEnd	exonCount	exonStarts	exonEnds	score	name2	cdsStartStat	cdsEndStat	exonFrames	kgID	description	ensemblID	tRnaName	spID	spDisplayID	protAcc	mRNA	rfamAcc

      0	NM_001376542	chr1	+	66999275	67216822	67000041	67208778	25	66999275,66999928,67091529,67098752,67105459,67108492,67109226,67126195,67133212,67136677,67137626,67138963,67142686,67145360,67147551,67154830,67155872,67161116,67184976,67194946,67199430,67205017,67206340,67206954,67208755,	66999620,67000051,67091593,67098777,67105516,67108547,67109402,67126207,67133224,67136702,67137678,67139049,67142779,67145435,67148052,67154958,67155999,67161176,67185088,67195102,67199563,67205220,67206405,67207119,67216822,	0	SGIP1	cmpl	cmpl	-1,0,1,2,0,0,1,0,0,0,1,2,1,1,1,1,0,1,1,2,2,0,2,1,1,	NA	NA	NA	NA	NA	NA	NA	NA	NA
      ```

    - _nearest_: A pre-calculated `gene` track that is intersected with a target `gene` track.

      Example:

      ```
      - name: refSeq.gene
        dist: false
        storeNearest: true
        to: txEnd
        type: nearest
        features:
        - name2
        from: txStart
        local_files:
        - hg19.kgXref.chr*.gz
      ```

      Options:

      - `dist`: bool
        - Calculate the distance to the nearest target gene record. If the

    - _vcf_: A VCF v4.\* file

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
