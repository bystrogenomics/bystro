# Bystro
<h2>Annotation Field Description</h2>
###### *Italicized fields* are custom Bystro fields. All others are sourced as described.

<br/>

<h5>General output information: </h5>
<p style="margin-left: 40px"> Missing data in the annotation is marked by <strong>'!'</strong>  </p>
<p style="margin-left: 40px"> Multiple transcripts are separated by <strong>';'</strong> </p>
<p style="margin-left: 40px"> Multiallelic sites are separated by <strong>'|'</strong></p>
<p style="margin-left: 40px"> Annotated output data is ordered in the same way as the original file. </p>

<br/>

<h5>Input fields</h5>
###### Sourced from the input file, or calculated based on input fields


**chrom** - chromosome 

**pos**  - genomic position

**type** - the type of variant
  * <md-icon class='material-icons small list-center'>help</md-icon> VCF format types: **SNP**, **INS**, **DEL**
  * <md-icon class='material-icons small list-center'>help</md-icon> SNP format types: **SNP**, **INS**, **DEL**, **MULTIALLELIC**, **DENOVO_***

<em>**discordant**</em> - does the input file's reference allele differ from Bystro's genome assembly? (1 if yes, 0 otherwise)

<em>**trTv**</em> - is the site a transition (1), transversion (2), or neither (0)?

**alt** - the alternate/nonreference allele
  * <md-icon class='material-icons small list-center'>help</md-icon> VCF multiallelics are split, one line each

<em>**heterozygotes**</em> - all samples that are heterozygotes for the alternate allele 

<em>**homozygotes**</em> - all samples that are homozygotes for the alternate allele 

<em>**missingGenos**</em> - all samples that have at least one '.' (VCF) or 'N' (SNP) genotype call.

  * <md-icon class='material-icons small list-center'>help</md-icon> **Note**: No samples are dropped

<br/>
##### Reference Assembly
###### Sourced from UCSC

**ref** - the reference allele
  * <md-icon class='material-icons small list-center'>help</md-icon> e.g Human (hg38, hg19), Mouse (mm10, mm9), Fly (dm6), C.elegans (ce11), etc.

<br/>
#####refSeq (<a href='https://www.ncbi.nlm.nih.gov/books/NBK50679/' target='_blank'>FAQ</a>)
###### Sourced from UCSC refGene (<a href='https://sc-bro.nhlbi.nih.gov/cgi-bin/hgTables?hgsid=554_JXUlabut7OUQtCyNphC8FGaeUJnj&hgta_doSchemaDb=hg38&hgta_doSchemaTable=refGene' target='blank'>schema</a>) and kgXref (<a href='https://sc-bro.nhlbi.nih.gov/cgi-bin/hgTables?hgsid=554_JXUlabut7OUQtCyNphC8FGaeUJnj&hgta_doSchemaDb=hg38&hgta_doSchemaTable=kgXref' target='_blank'>schema</a>)

All overlapping RefSeq transcripts are annotated (no prioritization, all possible values are reported)

<em>**refSeq.siteType**</em> - the effect the ```alt``` allele has on this transcript.
  * <md-icon class='material-icons small list-center'>help</md-icon> Possible types: **intronic**, **exonic**, **UTR3**, **UTR5**, **spliceAcceptor**, **spliceDonor**, **ncRNA**, **intergenic**
  * <md-icon class='material-icons small list-center'>help</md-icon> This is the only field that will have a value when a site is intergenic

<em>**refSeq.exonicAlleleFunction**</em> - The coding effect of the variant
  * <md-icon class='material-icons small list-center'>help</md-icon> Possible values: **synonymous**, **nonSynonymous**, **indel-nonFrameshift**, **indel-frameshift**, **stopGain**, **stopLoss**, **startLoss**

<em>**refSeq.refCodon**</em> - the codon based on *in silico* transcription of the reference assembly

<em>**refSeq.altCodon**</em> - the *in silico* transcribed codon after modification by the ```alt``` allele

<em>**refSeq.refAminoAcid**</em> - the amino acid based on *in silico* translation of the transcript

<em>**refSeq.altAminoAcid**</em> - the *in silico* translated amino acid after modification by the ```alt``` allele

<em>**refSeq.codonPosition**</em> - the site's position within the codon (1, 2, 3)

<em>**refSeq.codonNumber**</em> - the codon number within the transcript

**refSeq.strand** - the positive or negative watson/crick strand 

**refSeq.kgID** - UCSC's <a href='https://www.ncbi.nlm.nih.gov/pubmed/16500937' target='_blank'>Known Genes</a> ID

**refSeq.mRNA** - mRNA ID, the transcript ID starting with NM_

**refSeq.spID** - <a href="http://www.uniprot.org" target='_blank'>UniProt</a> protein accession number

**refSeq.spDisplayID** - <a href="http://www.uniprot.org" target='_blank'>UniProt</a> display ID

**refSeq.protAcc** - NCBI protein accession number

**refSeq.description** - long form description of the RefSeq transcript

**refSeq.rfamAcc** - <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC165453/" target="_blank">Rfam</a> accession number

**refSeq.name** - RefSeq transcript ID

**refSeq.name2** - RefSeq gene name

<br/>
##### *refSeq.nearest*
###### The nearest transcript(s), upstream or downstream for every position in the genome

<em>**refSeq.nearest.name**</em> - the nearest transcript(s) RefSeq transcript ID

<em>**refSeq.nearest.name2**</em> - the nearest transcript(s) RefSeq gene name

<br/>
##### *refSeq.clinvar*
###### Alleles found in Clinvar that are larger than 32bp and overlap a refSeq transcript

We report these separately because large alleles are less likely to be relevant to small snps and indels

Clinvar variants are reported based on position and **do not necessarily correspond to the input file's alleles at the same position**

<em>**refSeq.clinvar.alleleID**</em> - unique Clinvar identifier

<em>**refSeq.clinvar.phenotypeList**</em> - associated pheontypes

<em>**refSeq.clinvar.clinicalSignificance**</em> - designation of significance (i.e. benign, pathogenic, etc) from clinical reports

<em>**refSeq.clinvar.type**</em> - the variant type (i.e. single nucleotide variant)

<em>**refSeq.clinvar.origin**</em> - origin tissue for the clinical sample in which the variant was identified (not always provided)

<em>**refSeq.clinvar.numberSubmitters**</em> - total number of submissions of the Clinvar variant

<em>**refSeq.clinvar.reviewStatus**</em> - level of intepretation of the variant provided
  * <md-icon class='material-icons small list-center'>help</md-icon> Such as "reviewed by expert panel"

<em>**refSeq.clinvar.chromStart**</em> - chromosome start site for the clinvar record

<em>**refSeq.clinvar.chromEnd**</em> - chromosome end site for the clinvar record

<br/>
##### Genome-wide variant scores
###### Predications of conservation, evolution, and deleteriousness

<a target="__blank" href='http://compgen.cshl.edu/phast/background.php'>**phastCons**</a> - a conservation score that includes neighboring bases

<a target="__blank" href='http://compgen.cshl.edu/phast/background.php'>**phyloP**</a> - a conservation score that does not include neighboring bases

<a target="__blank" href='http://cadd.gs.washington.edu/'>**cadd**</a> - a score for the deleteriousness of a variant 

<br/>

##### dbSNP (<a target="__blank" href='https://www.ncbi.nlm.nih.gov/snp'>FAQ</a>)
###### The larget database of genetic variation

dbSNP variants up to **32 bases** in length are reported

dbSNP variants are reported based on position and **do not necessarily correspond to the input file's alleles at the same position**

**dbSNP.name**</a> - snp name, usually rs and a number

**dbSNP.strand** - strand orientation (+/-)

**dbSNP.observed** - observed SNP alleles at this position (+/- for indels)

**dbSNP.class** - variant type; includes single, insertion, and deletion

**dbSNP.func** - site type for the SNP name

**dbSNP.alleles** - SNP alleles in the dbSNP database

**dbSNP.alleleNs** - chromosome sample counts

**dbSNP.alleleFreqs** - major and minor allele frequencies

<br/>
##### Clinvar (<a href='https://www.ncbi.nlm.nih.gov/clinvar/docs/faq/' target='_blank'>FAQ</a>)
###### Clinically-reported human variants (hg38 and hg19 only)

Clinvar variants up to **32 bases** in length are reported

Clinvar variants are reported based on position and **do not necessarily correspond to the input file's alleles at the same position**

**clinvar.alleleID** - unique clinvar identifier for a particular variant

**clinvar.phenotypeList** - list of associated phenotypes for variants at this position, including indels up to 32bp in size

<a target="__blank" href='https://www.ncbi.nlm.nih.gov/clinvar/docs/clinsig/'>**clinvar.clinicalSignificance**</a> - designation of significance for a variant (i.e. benign, pathogenic, etc) from a clinical report

**clinvar.Type** - type of variant (i.e. single nucleotide variant

**clinvar.Origin** - origin tissue for clinical sample (not always provided)

**clinvar.numberSubmitters** - total number of submissions in clinvar overlapping this position, including indels up to 32bp in size

<a target="__blank" href='https://www.ncbi.nlm.nih.gov/clinvar/docs/variation_report/#review_status'>**clinvar.reviewStatus**</a> - level of intepretation of the variant provided

<a target="__blank" href='https://www.ncbi.nlm.nih.gov/clinvar/docs/variation_report/#allele'>**clinvar.referenceAllele**</a> - reference allele for this position in clinvar

**clinvar.alternateAllele** - alternate allele(s) for this position seen in clinvar

<br/>


6. heterozygotes
    - If the sample has the given allele in copy nunmber matching the # of chromosomes
    - In multiallelic case, phased on allele

### The Bystro configuration file

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

# Directories and Files
These describe where the Bystro database and any source files are located.

1. `files_dir` : Where our source files are located. This is only necessary when building a database

Ex:
```
files_dir: /path/to/files/
```

2. `database_dir` : Where the database is located. Required
```
database_dir: /path/to/database/
```

# Statistics configuration
The "statistics" key describes the bystro-stats module configuration.
