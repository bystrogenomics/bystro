import random

import numpy as np

import pyarrow as pa  # type: ignore
import pyarrow.compute as pc  # type: ignore
import pyarrow.dataset as ds  # type: ignore
import pandas as pd

import pytest

np.random.seed(0)  # noqa: NPY002
random.seed(0)

loci = [
    "chr2:8871342:C:A",
    "chr2:8877042:G:A",
    "chr2:8910987:A:T",
    "chr2:8916847:A:T",
    "chr2:8997300:A:-1",
    "chr22:51183255:A:G",
    "chr22:51183421:C:T",
    "chr22:51193227:T:G",
    "chr22:51208006:T:-2",
    "chr22:51215185:C:T",
]
data = {
    "locus": loci,
    "1805": np.random.randint(0, 3, size=len(loci), dtype=np.uint16),  # noqa: NPY002
    "1847": np.random.randint(0, 3, size=len(loci), dtype=np.uint16),  # noqa: NPY002
    "4805": np.random.randint(0, 3, size=len(loci), dtype=np.uint16),  # noqa: NPY002
}
table = pa.Table.from_pandas(pd.DataFrame(data))

weights = pd.Series(np.random.rand(len(loci)), index=loci)  # noqa: NPY002

# Extend weights with new entries
weights_with_missing = pd.concat(
    [weights, pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=["new1", "new2", "new3", "new4", "new5"])]
)


@pytest.fixture
def create_feather_file(tmp_path):
    # Use pytest's temporary directory feature
    file_path = str(tmp_path / "test_dosage.feather")

    # write an IPC/feather file
    with pa.OSFile(file_path, "wb") as sink, pa.RecordBatchFileWriter(sink, table.schema) as writer:
        writer.write_table(
            table,
        )

    return file_path


@pytest.fixture
def create_feather_file_with_multiple_batches(tmp_path, batches=5):
    # Use pytest's temporary directory feature
    file_path = str(tmp_path / "test_dosage.feather")

    # write an IPC/feather file
    weights = {}
    with pa.OSFile(file_path, "wb") as sink, pa.RecordBatchFileWriter(sink, table.schema) as writer:
        for i in range(batches):
            # generates a random set of 10 loci, taking chromosomes from chr1-22,
            # positions from 1-100000000, and alleles from ACGT
            loci = [
                f"chr{random.randint(1,22)}:{random.randint(1,100000000)}:{random.choice(['A','C','G','T'])}:{random.choice(['A','C','G','T'])}"
                for i in range(10)
            ]
            # generates a random set of 10 genotypes, taking values from 0-2, for 3 samples
            data = {
                "locus": loci,
                "1805": np.random.randint(0, 3, size=len(loci), dtype=np.uint16),  # noqa: NPY002
                "1847": np.random.randint(0, 3, size=len(loci), dtype=np.uint16),  # noqa: NPY002
                "4805": np.random.randint(0, 3, size=len(loci), dtype=np.uint16),  # noqa: NPY002
            }
            print(data)
            # write the batch of data to our feather file
            writer.write_table(pa.Table.from_pandas(pd.DataFrame(data)))

            weights.update({locus: random.random() for locus in loci})  # noqa: NPY002

    return file_path, pd.Series(weights)


def test_prs_calculation(create_feather_file):
    """
    Test PRS calculations with Pandas
    This will be more memory efficient than going through Pandas
    """
    file_path = create_feather_file

    # Read the Feather file into a PyArrow Table
    with pa.memory_map(file_path, "r") as source:
        table = pa.ipc.open_file(source).read_all()

    # Compute the weighted sum for each sample column
    prs_scores = {}
    for column in table.column_names:
        if column != "locus":
            # Multiply the column by the weights and sum
            weighted_column = pc.multiply(table.column(column), pa.array(weights))
            prs_scores[column] = pc.sum(weighted_column).as_py()

    assert len(prs_scores) == len(table.column_names) - 1  # excluding 'locus' column
    print(prs_scores)


def calculate_prs_in_pandas(file_path, weights):
    """
    Calculate PRS scores efficiently, in batches
    """
    # Read the Feather file into a Pandas DataFrame
    with pa.memory_map(file_path, "r") as source:
        loaded_arrays = pa.ipc.open_file(source).read_all()
        loaded_arrays = loaded_arrays.to_pandas()
        loaded_arrays = loaded_arrays.set_index("locus")

    # Simulate Polygenic Risk Score Calculation
    prs_scores = loaded_arrays.multiply(weights, axis="index").sum()

    return prs_scores.to_dict()


def test_prs_calculations_with_pandas(create_feather_file):
    """
    Test PRS calculations with Pandas
    This will be less memory efficient than operating on the Feather file directly
    """
    print("running test_prs_calculations_with_pandas")
    file_path = create_feather_file

    prs_scores = calculate_prs_in_pandas(file_path, weights)
    assert len(prs_scores) == len(table.column_names) - 1  # excluding 'locus' column
    print(prs_scores)


def test_prs_calculations_arrow_with_missing_values(create_feather_file):
    """
    Test PRS calculations when the genotype weights have loci not in the genotype dosage matrix
    """
    file_path = create_feather_file

    # Read the Feather file into a PyArrow Table
    with pa.memory_map(file_path, "r") as source:
        table = pa.ipc.open_file(source).read_all()

    # Compute the weighted sum for each sample column
    prs_scores = {}
    weights_subsetted_to_valid = None
    for column_name in table.column_names:
        valid_loci = []
        if column_name == "locus":
            valid_loci = table.column(column_name).to_pylist()
            weights_subsetted_to_valid = weights_with_missing.loc[valid_loci]

        if column_name != "locus":
            sample_column = table.column(column_name)
            weighted_values = pc.multiply(sample_column, pa.array(weights_subsetted_to_valid))
            prs_scores[column_name] = pc.sum(weighted_values).as_py()

    assert len(prs_scores) == len(table.column_names) - 1  # excluding 'locus' column
    print(prs_scores)


def calculate_prs_in_batches(file_path, weights):
    """
    Calculate PRS scores efficiently, in batches
    """

    my_dataset = ds.dataset([file_path], format="arrow")
    mask = pc.field("locus").isin(weights.index)
    my_dataset = my_dataset.filter(mask)

    # Read the Feather file into a Pandas DataFrame in batches
    # and calculate PRS in batches
    prs_scores: dict[str, float] = {}
    for batch in my_dataset.to_batches():
        loaded_arrays = batch.to_pandas()
        loaded_arrays = loaded_arrays.set_index("locus")
        weights_subset = weights.loc[loaded_arrays.index]
        prs_scores_batch = loaded_arrays.multiply(weights_subset, axis="index").sum()
        # add the batch scores to the total scores for each sample
        for sample in prs_scores_batch.index:
            if sample in prs_scores:
                prs_scores[sample] += prs_scores_batch[sample]
            else:
                prs_scores[sample] = prs_scores_batch[sample]

    return prs_scores


def test_prs_one_batch(create_feather_file):
    """
    Test PRS calculations with Pandas
    This will be less memory efficient than operating on the Feather file directly
    """
    file_path = create_feather_file
    prs_scores = calculate_prs_in_batches(file_path, weights)
    prs_scores_pandas = calculate_prs_in_pandas(file_path, weights)

    # deep equality check that these two dictionaries contain the same keys and values
    assert prs_scores == prs_scores_pandas

    assert len(prs_scores) == 3
    for sample in prs_scores:
        assert prs_scores[sample] > 0

    print(prs_scores)


def test_prs_multiple_batches(create_feather_file_with_multiple_batches):
    """
    Test PRS calculations with Pandas
    This will be less memory efficient than operating on the Feather file directly
    """
    file_path, weights = create_feather_file_with_multiple_batches
    prs_scores = calculate_prs_in_batches(file_path, weights)

    assert len(prs_scores) == 3
    for sample in prs_scores:
        assert prs_scores[sample] > 0

    print(prs_scores)


def test_prs_columnar_one_batch(create_feather_file_with_multiple_batches):
    file_path, weights = create_feather_file_with_multiple_batches

    expected = calculate_prs_in_pandas(file_path, weights)

    mask = pc.field("locus").isin(weights.index)

    my_dataset = ds.dataset([file_path], format="arrow")
    samples = set(my_dataset.schema.names) - set(["locus"])

    filtered_data = my_dataset.filter(mask)

    prs_scores = {}
    for sample in samples:
        sample_genotypes = filtered_data.to_table(["locus", sample]).to_pandas()
        sample_genotypes = sample_genotypes.set_index("locus")

        weights_subset = weights.loc[sample_genotypes.index]
        prs_scores_batch = sample_genotypes.multiply(weights_subset, axis="index").sum()
        prs_scores[sample] = prs_scores_batch.item()

    assert prs_scores == expected
    print(prs_scores)


def test_prs_columnar_batches(create_feather_file_with_multiple_batches):
    file_path, weights = create_feather_file_with_multiple_batches
    mask = pc.field("locus").isin(weights.index)

    my_dataset = ds.dataset([file_path], format="arrow")
    samples = set(my_dataset.schema.names) - set(["locus"])

    filtered_data = my_dataset.filter(mask)

    prs_scores: dict[str, float] = {}
    for batch in filtered_data.to_batches():
        for sample in samples:
            sample_genotypes = batch.select(["locus", sample]).to_pandas()
            sample_genotypes = sample_genotypes.set_index("locus")

            weights_subset = weights.loc[sample_genotypes.index]
            prs_scores_batch = sample_genotypes.multiply(weights_subset, axis="index").sum()

            if sample in prs_scores:
                prs_scores[sample] += prs_scores_batch
            else:
                prs_scores[sample] = prs_scores_batch

    print(prs_scores)
