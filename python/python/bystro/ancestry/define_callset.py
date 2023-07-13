import re
from pathlib import Path

import pandas as pd
import tqdm
from liftover import get_lifter
from train import DATA_DIR, ANCESTRY_DIR, INTERMEDIATE_DATA_DIR

converter = get_lifter("hg19", "hg38")

ILLUMINA_FILEPATH = DATA_DIR / "Human660W-Quad_v1_H.csv"
import sys


def get_watson_crick_complement(b):
    wc_dict = {"A": "T", "T": "A", "G": "C", "C": "G"}
    return wc_dict[b]


def load_illumina_callset():
    illumina_df = pd.read_csv(ILLUMINA_FILEPATH, skiprows=7)
    assert set(illumina_df.GenomeBuild.dropna()) == {37.1}
    variants = []
    liftover_attempts = 0
    liftover_failures = 0
    for _i, row in tqdm.tqdm(illumina_df.iterrows(), total=len(illumina_df)):
        chromosome = str(row.Chr)
        if chromosome in ["X", "Y", "0", "MT", "XY"]:
            continue
        try:
            position = str(int(row.MapInfo))
        except:
            continue
        liftover_attempts += 1
        try:
            results = converter[int(chromosome)][int(position)]
            assert len(results) == 1
            chromosome38, position38, strand38 = results[0]
            position38 = str(position38)
        except:
            print("couldn't convert:", chromosome, position)
            liftover_failures += 1
            continue
        if match := re.match(r"\[([ACGT])/([ACGT])\]", str(row.SNP)):
            allele1, allele2 = match.groups()
        else:
            continue
        strand = row.RefStrand
        if strand == "-":
            allele1 = get_watson_crick_complement(allele1)
            allele2 = get_watson_crick_complement(allele2)

        variants.append(":".join([chromosome38, position38, allele1, allele2]))
    liftover_failure_rate_pct = liftover_failures / liftover_attempts * 100
    print(
        f"liftover failures: {liftover_failure_rate_pct:.3f}%",
    )
    return variants


def load_affymetrix_callset():
    chip_df = pd.read_csv(
        DATA_DIR / "Axiom_PMRA.na35.annot.csv",
        comment="#",
        index_col=0,
    )
    assert set(chip_df.Strand) == {"+"}
    assert set(chip_df["Physical Position"] - chip_df["Position End"]) == {0}
    variants = []
    for i, row in chip_df.iterrows():
        variant = ":".join(
            [
                "chr" + str(row.Chromosome),
                str(row["Physical Position"]),
                row["Ref Allele"],
                row["Alt Allele"],
            ]
        )

        variants.append(variant)
    variants38 = [get_38_from_37(v) for v in variants]
    chip_df["variant"] = variants38
    return chip_df.variant


def compare_illumina_affy_variants():
    illumina_variants = load_illumina_callset()
    affy_variants = load_affy_callset()
    shared_variants = pd.DataFrame(sorted(set(illumina_variants).intersection(affy_variants)))
    shared_variants.to_csv(
        INTERMEDIATE_DATA_DIR / "shared_illumina_affy_variants.csv", index=False, header=False
    )
    return shared_variants
