# Bystro Annotation Field Description
<h4> <em>Italicized fields</em> are custom Bystro fields. All others are sourced as described.</h4>

<br/>

### General output information:
<p style="margin-left: 40px"> Missing data in the annotation is marked by <strong>'NA'</strong>  </p>
<p style="margin-left: 40px"> Multiple values for a single annotated position are separated by <strong>';'</strong> </p>
<p style="margin-left: 40px"> Multiple positions on a single annotation line (occurs with indels only) are separated by <strong>'|'</strong> </p>
<p style="margin-left: 40px"> Annotated output data is ordered in the same way as the original file. </p>

Reserved characters:
  - ";" "|" "/"
  - "/" Will be used in a future release to denote overlapping data from a single track
    - For instance if 2 different dbSNP records overlap, which often occurs with indels, or when two refSeq transcripts overlap at the same position
    - Currently such sites are compressed to ";", but this loses information when a 1:1 relationship does not exist between a track's fields
      - For instance dbSNP.alleles are in the form Major;Minor1;Minor2 and dbSNP.name may or may not be a single value, regardless of # of minor alleles
      - When multiple dbSNP rows overlap, we store each field at that position in a 1D array, which loses the relationship between dbSNP.alleles and dbSNP.name
<br/>

### Input fields
<h4>Sourced from the input file, or calculated based on input fields</h4>


**chrom** - chromosome 

**pos**  - genomic position

**type** - the type of variant
  * VCF format types: **SNP**, **INS**, **DEL**, **MULTIALLELIC**
  * SNP format types: **SNP**, **INS**, **DEL**, **MULTIALLELIC**, **DENOVO_***
  
<em>**discordant**</em> - does the input file's reference allele differ from Bystro's genome assembly? (1 if yes, 0 otherwise)

<em>**trTv**</em> - is the site a transition (1), transversion (2), or neither (0)?

**alt** - the alternate/nonreference allele
  * VCF multiallelics are split, one line each

<em>**heterozygotes**</em> - all samples that are heterozygotes for the alternate allele 

<em>**homozygotes**</em> - all samples that are homozygotes for the alternate allele 

<em>**missingGenos**</em> - all samples that have at least one '.' (VCF) or 'N' (SNP) genotype call.

  * **Note**: No samples are dropped

Multiallelic variants are always decomposed into bi-allelic variants on separate lines, and given the type **MULTIALLELIC**
  * Heterozygotes/Homozygotes are called based on the number of alleles for a given decomposed variants
    * For instance, if the variant is pos:1 alt:A,C ref:T and Sample1 is 1/1 on line 1: pos:1 alt:A ref:T hets:Sample1 and on line 2: pos:1 alt:C ref:T hets:Sample1 
<br/>

### Reference Assembly
<h4>Sourced from UCSC</h4>

**ref** - the reference allele
  * e.g Human (hg38, hg19), Mouse (mm10, mm9), Fly (dm6), C.elegans (ce11), etc.

<br/>

### refSeq (<a href='https://www.ncbi.nlm.nih.gov/books/NBK50679/' target='_blank'>FAQ</a>)
<h4>Sourced from UCSC refGene (<a href='https://sc-bro.nhlbi.nih.gov/cgi-bin/hgTables?hgsid=554_JXUlabut7OUQtCyNphC8FGaeUJnj&hgta_doSchemaDb=hg38&hgta_doSchemaTable=refGene' target='blank'>schema</a>) and kgXref (<a href='https://sc-bro.nhlbi.nih.gov/cgi-bin/hgTables?hgsid=554_JXUlabut7OUQtCyNphC8FGaeUJnj&hgta_doSchemaDb=hg38&hgta_doSchemaTable=kgXref' target='_blank'>schema</a>)</h4>

All overlapping RefSeq transcripts are annotated (no prioritization, all possible values are reported)

<em>**refSeq.siteType**</em> - the effect the ```alt``` allele has on this transcript.
  * Possible types: **intronic**, **exonic**, **UTR3**, **UTR5**, **spliceAcceptor**, **spliceDonor**, **ncRNA**, **intergenic**
  * This is the only field that will have a value when a site is intergenic

<em>**refSeq.exonicAlleleFunction**</em> - The coding effect of the variant
  * Possible values: **synonymous**, **nonSynonymous**, **indel-nonFrameshift**, **indel-frameshift**, **stopGain**, **stopLoss**, **startLoss**

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

### refSeq.nearest
<h4> The nearest transcript(s), upstream or downstream for every position in the genome</h4>

<em>**refSeq.nearest.name**</em> - the nearest transcript(s) RefSeq transcript ID

<em>**refSeq.nearest.name2**</em> - the nearest transcript(s) RefSeq gene name

<br/>

### refSeq.clinvar
<h4> Alleles found in Clinvar that are larger than 32bp and overlap a refSeq transcript </h4>

We report these separately because large alleles are less likely to be relevant to small snps and indels

Clinvar variants are reported based on position and **do not necessarily correspond to the input file's alleles at the same position**

<em>**refSeq.clinvar.alleleID**</em> - unique Clinvar identifier

<em>**refSeq.clinvar.phenotypeList**</em> - associated pheontypes

<em>**refSeq.clinvar.clinicalSignificance**</em> - designation of significance (i.e. benign, pathogenic, etc) from clinical reports

<em>**refSeq.clinvar.type**</em> - the variant type (i.e. single nucleotide variant)

<em>**refSeq.clinvar.origin**</em> - origin tissue for the clinical sample in which the variant was identified (not always provided)

<em>**refSeq.clinvar.numberSubmitters**</em> - total number of submissions of the Clinvar variant

<em>**refSeq.clinvar.reviewStatus**</em> - level of intepretation of the variant provided
  * Such as "reviewed by expert panel"

<em>**refSeq.clinvar.chromStart**</em> - chromosome start site for the clinvar record

<em>**refSeq.clinvar.chromEnd**</em> - chromosome end site for the clinvar record

<br/>

### Genome-wide variant scores
<h4> Predications of conservation, evolution, and deleteriousness </h4>

<a target="__blank" href='http://compgen.cshl.edu/phast/background.php'>**phastCons**</a> - a conservation score that includes neighboring bases

<a target="__blank" href='http://compgen.cshl.edu/phast/background.php'>**phyloP**</a> - a conservation score that does not include neighboring bases

<a target="__blank" href='http://cadd.gs.washington.edu/'>**cadd**</a> - a score for the deleteriousness of a variant 

<br/>

### dbSNP (<a target="__blank" href='https://www.ncbi.nlm.nih.gov/snp'>FAQ</a>)
<h4> The larget database of genetic variation </h4>

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

### Clinvar (<a href='https://www.ncbi.nlm.nih.gov/clinvar/docs/faq/' target='_blank'>FAQ</a>)
<h4> Clinically-reported human variants (hg38 and hg19 only) </h4>

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
