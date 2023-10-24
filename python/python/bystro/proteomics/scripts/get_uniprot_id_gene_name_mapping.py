#!/usr/bin/env python

"""Download a list of uniprot_id / gene_name pairs from Uniprot and store the result on disk as csv."""

import pandas as pd
import tqdm
from unipressed import UniprotkbClient  # type: ignore[import]

from bystro.utils.config import BYSTRO_PROJECT_ROOT


def get_uniprot_id_gene_name_mapping() -> pd.DataFrame:
    """Get a dataframe associating each uniprot id with its cognate gene names."""
    APPROXIMATE_TOTAL_RECORDS = 240_000
    query_results = UniprotkbClient.search(
        query="(organism_name:homo)",
    )
    uniprot_id_gene_name_mapping = []
    for record in tqdm.trange(query_results.each_record(), total=APPROXIMATE_TOTAL_RECORDS):
        genes = record.get("genes", [])  # type: ignore[attr-defined]
        gene_names = []
        for gene in genes:
            if (gene_name := gene.get("geneName")) and (value := gene_name.get("value")):
                gene_names.append(value)
        if not gene_names:
            gene_names = [None]
        for gene_name in gene_names:
            uniprot_id_gene_name_mapping.append(
                (record["primaryAccession"], gene_name)  # type: ignore[index]
            )
    uniprot_id_gene_name_mapping_df = pd.DataFrame(
        uniprot_id_gene_name_mapping, columns=["uniprot_accession", "gene_name"]
    )
    return uniprot_id_gene_name_mapping_df


if __name__ == "__main__":
    uniprot_id_gene_name_mapping_df = get_uniprot_id_gene_name_mapping()
    output_filename = (
        BYSTRO_PROJECT_ROOT / "python/python/bystro/proteomics" / "uniprot_id_gene_name_mapping.csv"
    )
    uniprot_id_gene_name_mapping_df.to_csv(output_filename, index=False)
