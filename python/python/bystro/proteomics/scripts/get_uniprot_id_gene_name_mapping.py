#!/usr/bin/env python

"""Download a list of uniprot_id / gene_name pairs from Uniprot and store the result on disk as csv."""

import logging
import pandas as pd
import tqdm
from unipressed import UniprotkbClient  # type: ignore[import]

from bystro.utils.config import BYSTRO_PROJECT_ROOT

logger = logging.getLogger(__file__)


def get_uniprot_id_gene_name_mapping() -> pd.DataFrame:
    """Get a dataframe associating each uniprot id with its cognate gene names."""
    # TODO: currently we only query for human proteins, but this
    # should later to be expanded to include (at least) all the
    # reference genomes supported in bystro/config.
    query_results = UniprotkbClient.search(
        query="(organism_name:homo)",
    )
    uniprot_id_gene_name_mapping = []
    for record in tqdm.tqdm(query_results.each_record()):
        uniprot_id = record["primaryAccession"]
        genes = record.get("genes", [])  # type: ignore[attr-defined]
        gene_names = []
        for gene in genes:
            if (gene_name := gene.get("geneName")) and (value := gene_name.get("value")):
                gene_names.append(value)
        # TODO: if gene_names is empty here, we won't record the
        # uniprot_id at all.  This is ok for now, but we may later
        # wish to revise this decision.
        for gene_name in gene_names:
            uniprot_id_gene_name_mapping.append((uniprot_id, gene_name))  # type: ignore[index]
    uniprot_id_gene_name_mapping_df = pd.DataFrame(
        uniprot_id_gene_name_mapping, columns=["uniprot_accession", "gene_name"]
    )
    return uniprot_id_gene_name_mapping_df


if __name__ == "__main__":
    logger.info("Downloading Uniprot ID / gene name data...")
    uniprot_id_gene_name_mapping_df = get_uniprot_id_gene_name_mapping()
    logger.info("Download complete")
    output_filename = (
        BYSTRO_PROJECT_ROOT / "python/python/bystro/proteomics" / "uniprot_id_gene_name_mapping.csv"
    )
    uniprot_id_gene_name_mapping_df.to_csv(output_filename, index=False)
