# Proteomics README
This module provides functionality for analyzing proteomics datasets, principally from Fragpipe
software suite, in conjunction with genomic annotations provided by Bystro.

## Proteomics Datasets
The fundamental entity in a proteomics dataset is an _abundance matrix_, which records the measured
protein abundances of a set of proteins in a set of samples.  Also included in the dataset is an
_annotation matrix_, which records metadata pertaining to the samples.  Such datasets are
represented within Bystro by the class `TandemMassTagDataset` in `fragpipe_tandem_mass_tag.py`.
(The file `fragpipe_data_independent_analysis.py` provides similar functionality for Fragpipe DIA
datasets.  Although there are many proteomics protocols with their own file formats, they are
largely `sui generis` and we will constrain our discussion to Fragpipe TMT datasets below.  For
further details see the [Fragpipe
documentation](https://fragpipe.nesvilab.org/docs/tutorial_fragpipe_outputs.html)).

## CLI
The file `proteomics_cli.py` provides a CLI for uploading proteomics datasets to Bystro.  AFter
authentication, the user may upload a dataset via the command:

```
python proteomics_cli.py upload-proteomics-dataset --protein-abundance-file PROTEIN_ABUNDANCE_FILE \
                                                   --experiment-annotation-file EXPERIMENT_ANNOTATION_FILE
                                                   [--dir DIR]
```
where `DIR` is an optional location for storage within Bystro.

## Proteomics Listener
The loading of proteomics datasets within Bystro is handled by the proteomics listener, a python
service that is initialized during Bystro startup and listens to the `proteomics` beanstalkd tube
for incoming `ProteomicsSubmission` messages.  Upon receipt of a `ProteomicsSubmission` message, the
listener loads the file described in the submission payload and returns the result in a
`ProteomicsResponse`.

## Annotation Interface
It's often desirable to analyze a proteomics dataset in conjunction with a genomic annotation file
(of the same subjects) provided by Bystro.  The basic workflow of this joint analysis is as follows:

1.  The user queries the annotation file with an arbitrary OpenSearch query in order to select a
    subset of rows (i.e. variant records) from the annotation file.  For example, a user might
    provide the query: "exonic (gnomad.genomes.af:<0.1 || gnomad.exomes.af:<0.1)" which means
    "return all exonic variants where the allele frequency in gnomad.genomes or gnomad.exomes is
    less than 10%".
2.  The annotation interface returns a list of variant / sample pairs, i.e. for every valid variant
    and every sample containing that variant, we return a record of the form: `(sample_id, chrom,
    pos, ref, alt, gene_name, dosage)`.  This functionality is provided through the method
    `get_annotation_result_from_query`.
3.  The annotation query results are then joined to a given proteomics dataset on `sample_id` and
    `gene_name`.  That is to say, the result will be a Pandas dataframe with the columns:
    `(sample_id, chrom, pos, ref, alt, gene_name, dosage, protein_abundance)`, where
    `protein_abundance` is the abundance of protein `gene_name` for sample `sample_id`.  This
    functionality is provided through the method `join_annotation_result_to_proteomics_dataset`.
	
Notes:
1.  In general, a given subject's `sample_id` in the genomic annotation file may not be identical to
    its `sample_id` in the proteomic dataset.  Instead, a subject may globally identified through a
    `tracking_id`.  In that case, the user may provide two mappings:
    `get_tracking_id_from_genomic_sample_id` and `get_tracking_id_from_proteomic_sample_id`, and
    these helper functions will be used to canonicalize the `sample_ids` before joining the two
    datasets.

2.  We assume (in accordance with the datasets we have seen so far) that Fragpipe will map peptide
    Uniprot IDs to HUGO gene name symbols, so that annotation gene names can be identified with
    proteomics gene names in that namespace, but this may not always be the case.  Functionality to
    map from Uniprot IDs to HUGO symbols and vice versa is described below.
	
## Uniprot / HUGO symbol mapping
	An additional module, `uniprot_id_gene_name_mapping.py`, is provided to convert Uniprot IDs to
    gene names and vice versa.  This module provides two public methods,
    `get_gene_names_from_uniprot_id` and `get_uniprot_ids_from_gene_name`.  (NB: this mapping is in
    general many to many, hence each returns a list of cognates in the other namespace.)  These
    mappings are populated by a data download from Uniprot, which can be run by the script
    `scripts/get_uniprot_id_gene_name_mapping.py`.  Currently, this script only queries the human
    proteome, but can be trivially amended to include all model organisms supported in
    `bystro/config/*.mapping.yml`, or indeed all of Uniprot.





