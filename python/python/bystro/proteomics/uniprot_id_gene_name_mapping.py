"""Provide functions to convert Uniprot IDs to gene names and vice versa."""

import pandas as pd

from bystro.utils.config import BYSTRO_PROJECT_ROOT

pd.options.future.infer_string = True  # type: ignore

_MAPPING_FILENAME = (
    BYSTRO_PROJECT_ROOT / "python/python/bystro/proteomics" / "uniprot_id_gene_name_mapping.csv"
)
try:
    _UNIPROT_ID_GENE_NAME_MAPPING = pd.read_csv(_MAPPING_FILENAME)
except FileNotFoundError as e:
    err_msg = (
        "Uniprot ID / gene name mapping file not found: "
        "run scripts/get_uniprot_id_gene_name_mapping.py to create it"
    )
    raise FileNotFoundError(err_msg) from e


def get_gene_names_from_uniprot_id(uniprot_id: str) -> list[str]:
    """Return a list of gene names associated with the given Uniprot ID."""
    uniprot_idx = _UNIPROT_ID_GENE_NAME_MAPPING.uniprot_accession == uniprot_id
    if not uniprot_idx.sum():
        err_msg = f"Couldn't find Uniprot ID {uniprot_id} in mapping"
        raise ValueError(err_msg)
    gene_names = _UNIPROT_ID_GENE_NAME_MAPPING[uniprot_idx].gene_name
    return [gene_name for gene_name in gene_names if pd.notna(gene_name)]


def get_uniprot_ids_from_gene_name(gene_name: str) -> list[str]:
    """Return a list of Uniprot IDs associated with the given gene name."""
    gene_name_idx = _UNIPROT_ID_GENE_NAME_MAPPING.gene_name == gene_name
    if not gene_name_idx.sum():
        err_msg = f"Couldn't find gene name {gene_name} in gene name"
        raise ValueError(err_msg)
    uniprot_ids = _UNIPROT_ID_GENE_NAME_MAPPING[gene_name_idx].uniprot_accession
    return list(uniprot_ids)
