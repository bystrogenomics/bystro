# Annotation Fields (Human Assembly hg38 and hg19)

<h2>Annotation Field Description</h2>

##### _Italicized fields_ are custom Bystro fields. All others are sourced as described.

<br/>

#### Summary of the Bystro Annotator

Bystro takes 1+ VCF or SNP (PEMapper/PECaller) files, performs quality control, generates statistics, and outputs an annotated file that is in an easy-to-parse tab-separated format.

Each row in the output file corresponds to a single variant site, and contains a set of annotations that describe the site's genomic context, functional impact, allele frequencies in various populations, and more.

The annotations are divided into several categories, each of which is described in detail below.

#### Output Data Structures

Bystro's output format is designed to retain and reflect complex relationships within genomic annotations. Here are key aspects of how we store the data:

1. **Array-Based Fields**: Each annotation field is an array. For fields with multiple values (e.g., transcripts, gene names), the values are separated by semicolons (`;`). The order of the values is maintained across related fields to preserve relationships between the data points. For example:

   - `refSeq.name` might contain `NM1;NM2`, and `refSeq.name2` (gene symbols for these transcripts) could be `Gene1;Gene1`. This ensures the first transcript, `NM1`, corresponds to `Gene1`, and the second transcript, `NM2`, also corresponds to `Gene1`. These relationships are maintained across all fields within a track.

2. **Nested Arrays**: Some fields may be nested arrays, where the individual entries are further divided using forward slashes (`/`). For example, if a transcript has alternate IDs, you may see `NM1a/NM1b;NM2`, indicating two alternate IDs for the first transcript. This way we can maintain the relationships between the order of fields.

3. **Insertions and Deletions**: For transcript-level annotations like `refSeq.*` (refSeq transcript annotations), `nearest.*`, and `nearestTss.*`, insertions and deletions affecting multiple nucleotides are separated by pipes (`|`). This allows reporting a transcript consequence per disrupted base.

4. **Reserved Delimiters**: Reserved delimiters like `;`, `/`, and `|` are essential for maintaining the structure of relationships in the data. If these characters appear in the source data, they are replaced with commas to ensure they can still be used effectively as delimiters in the output.

This structure enables precise retention of complex data relationships across multiple annotation fields while maintaining a highly structured and parseable output.

#### What Information Can Bystro Output?

Bystro is a general-purpose annotation engine, and has no restrictions on the types of annotations/feature labels it can output. The annotations are sourced from the Bystro Database chosen, and is described based on a YAML configuration file, which describes the fields to be output, and the sources of the annotations.

A Bystro Database is a high-performance key-value database that uses the Lightning Memory Map Database (LMDB) engine. It supports millions of lookups per second on a single machine, and can be used to store and retrieve annotations for millions of variants.

Bystro Databases can be re-created from the YAML configuration file corresponding to that database, and new databases with different information can be created by editing the YAML configuration file, and re-running the Bystro Database creation process.

#### Variant Representation

Bystro's variant representation deviates slightly from the standard VCF format. Namely, Bystro ensures that the reference base is always a single nuleotide, and does the appropriate normalization/transformation of input variants to ensure this. This variant representation greatly simplifies downstream analysis.

Compared with the VCF format, Bystro's variant representation has the following differences:

SNPs: Unchanged

Insertions: The reference base is always the base just before the insertion, and the alternate allele is the inserted sequence, preceeded by a +. The position is the first affected base.

- For instance, if the VCF had POS: 1 REF: AT and ALT: ACCT, Bystro will represent this as inputRef: T, alt: +CCT, pos: 1, with the position the same as that in the VCF file, which is the first position of the reference base (the base just before the first inserted base). In Bystro's case there is only 1 reference base always, so the position is that of the only affected base.

Deletions: VCFs require ALT alleles to be at least 1 nucleotide long, which means that to represent a deletion, the VCF has to "pad" the reference. Bystro removes this padding, and represents deletions as the reference base, followed by a - and the number of deleted bases. The position is the first affected base.

- For instance if the VCF had POS:1 REF: AT and ALT: A, Bystro will represent this as inputRef: T, alt: -1, pos: 2, notably shifting the position of the allele up by 1 base, since the deletion did not actually occur at position 1, but the next base over.

Multiallelics: Bystro decomposes multiallelic sites into separate rows, each with a single alternate allele. This ensures that each row has a single alternate allele, and that the annotations are accurate for each allele. The appropriate transformations are done to ensure that VCF padding rules are correctly transformed into Bystro's representation, and ensures that all variants outtputted in Bystro are left-aligned.

<br/>

### Annotation Fields for Default Human Assembly hg38 and hg19 Bystro Databases

#### Basic Fields

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

`ref` - The Bystro-annotated reference base(s), from the 'ref' track in the Bystro Database

- In the default Bystro Database, this is sourced from the UCSC reference genome
- In custom Bystro Databases, this can be sourced from any reference genome
- In the case of insertions the `ref` will be 2 bases long, the base just before the insertion, and the one right after
- In the case of deletions, the ref will be as long as the deletion, up to 32 bases (after that, the ref will be truncated)

<br/>

#### Transcript Annotations

In the default Bystro Database, we source transcript annotations from the UCSC refGene track, joined on other UCSC tracks: knownToEnsembl, kgXref, knownCanonical.

- See [refGene]('https://sc-bro.nhlbi.nih.gov/cgi-bin/hgTables?hgsid=554_JXUlabut7OUQtCyNphC8FGaeUJnj&hgta_doSchemaDb=hg38&hgta_doSchemaTable=refGene') and [kgXref]('https://sc-bro.nhlbi.nih.gov/cgi-bin/hgTables?hgsid=554_JXUlabut7OUQtCyNphC8FGaeUJnj&hgta_doSchemaDb=hg38& hgta_doSchemaTable=kgXref') for more information

- In custom Bystro Databases, these annotations can be sourced from any UCSC transcript track, and multiple such `gene` type tracks can be defined in a single Bystro Database (annotations for all will be outputted)

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

#### nearest.refSeq

The nearest transcript(s), calculated by trascript start, transcript end boundaries. Transcripts that are equidistant are all outputted.

`nearest.refSeq.name` - The nearest transcript(s) RefSeq transcript ID

`nearest.refSeq.name2` - The nearest transcript(s) RefSeq gene symbol

`nearest.refSeq.dist` - The distance to these transcripts. Negative values indicate the site is downstream of the transcript

<br/>

#### nearestTss.refSeq

The nearest transcript(s), calculated by the distance to the nearest transcript start site (TSS). Transcripts with the same TSS are all outputted.

`nearestTss.refSeq.name` - The nearest transcript(s) RefSeq transcript ID

`nearestTss.refSeq.name2` - The nearest transcript(s) RefSeq gene symbol

`nearestTss.refSeq.dist` - A single value; the distance to these transcripts' transcription start site. Negative values indicate the site is downstream of the TSS

<br/>

#### gnomAD Annotations

Annotations from the gnomAD v4.1 (hg38 assembly annotations) or v2.1.1 (hg19 assembly annotations) whole-genome set

Since the data available for hg19 and hg38 differ, we will discuss them separately below.

#### hg38 gnomad.joint

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

#### hg38 gnomad.genomes

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

#### hg38 gnomad.exomes

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

#### hg19 gnomad.genomes (v2.1.1 - latest release for hg19)

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

#### hg19 gnomad.exomes (v2.1.1 - latest release for hg19)

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

#### [dbSNP](https://www.ncbi.nlm.nih.gov/snp)

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

#### [cadd](http://cadd.gs.washington.edu)

A score >=0 that indicates deleteriousness of a variant. Variants with cadd > 15 are more likely to be deleterious.
See http://cadd.gs.washington.edu.

<br/>

#### [caddIndel](http://cadd.gs.washington.edu)

A score >=0 that indicates deleteriousness of a variant. Variants with cadd > 15 are more likely to be deleterious.
See http://cadd.gs.washington.edu.

caddIndel scores are only defined for indels and MNPs. For SNPs, use the `cadd` field.

- Note that because Bystro decomposes MNPs into "SNP" records, the `caddIndel` field will occasionally be populated for SNPs (which are in fact part of MNPs in the CADD Indel dataset).

`caddIndel.alt`: The Bystro VCF-preprocessor's ALT record for this CADD site. This should always match the row's `alt` field value

`caddIndel.id`: The CADD VCF `ID`

`caddIndel.PHRED`: The CADD PHRED score for the insertion or deletion

<br/>

#### clinvarVcf

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

// add logofunc.id logofunc.alt logofunc.prediction logofunc.neutral logofunc.gof logofunc.lof genebass.id genebass.alt genebass.phenocode genebass.description

<br/>

#### (hg38-only) [LoGoFunc](

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10688473/)
A machine learning method for predicting pathogenic GOF, pathogenic LOF, and neutral genetic variants, trained on a broad range of gene-, protein-, and variant-level features describing diverse biological characteristics.

`logofunc.id`: The LoGoFunc VCF `ID`

`logofunc.alt`: The Bystro VCF-preprocessor's ALT record for this LoGoFunc site. This should always match the row's `alt` field value

`logofunc.prediction`: The LoGoFunc prediction

`logofunc.neutral`: The LoGoFunc neutral score

`logofunc.gof`: The LoGoFunc gain of function (GOF) score

`logofunc.lof`: The LoGoFunc loss of function (LOF) score

#### (hg38-only) [GeneBass](<https://www.cell.com/cell-genomics/pdf/S2666-979X(22)00110-0.pdf>)

`genebass.id`: The GeneBass VCF `ID`

`genebass.alt`: The Bystro VCF-preprocessor's ALT record for this GeneBass site. This should always match the row's `alt` field value

`genebass.phenocode`: The GeneBass phenotype code

`genebass.description: The GeneBass description
