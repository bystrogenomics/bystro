import pytest

from bystro.proteomics.uniprot_id_gene_name_mapping import (
    get_uniprot_ids_from_gene_name,
    get_gene_names_from_uniprot_id,
)


def test_get_uniprot_ids_from_gene_name():
    gene_name = "MTOR"
    expected_uniprot_ids = [
        "P42345",
        "A0A8V8TRG9",
        "A0A8V8TQ52",
        "A0A8V8TQM6",
        "A0A8V8TQP2",
        "A0A8V8TQN3",
        "A0A8V8TR74",
    ]
    assert expected_uniprot_ids == get_uniprot_ids_from_gene_name(gene_name)


def test_get_uniprot_ids_from_gene_name_bad_input():
    with pytest.raises(ValueError, match="Couldn't find gene name FOO"):
        get_uniprot_ids_from_gene_name("FOO")


def test_get_gene_names_from_uniprot_ids():
    uniprot_id = "P42345"
    expected_gene_names = ["MTOR"]
    assert expected_gene_names == get_gene_names_from_uniprot_id(uniprot_id)


def test_get_gene_names_from_uniprot_id_bad_input():
    with pytest.raises(ValueError, match="Couldn't find Uniprot ID FOO"):
        get_gene_names_from_uniprot_id("FOO")
