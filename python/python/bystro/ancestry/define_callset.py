"""Calculate intersection of Illumina and Affymetrix chips."""

import logging
import re

import pandas as pd
import tqdm
from liftover import get_lifter

from bystro.ancestry.asserts import assert_equals, assert_true
from bystro.ancestry.train import DATA_DIR, INTERMEDIATE_DATA_DIR

logger = logging.getLogger(__name__)
converter = get_lifter("hg19", "hg38")

ILLUMINA_FILEPATH = DATA_DIR / "Human660W-Quad_v1_H.csv"
AFFYMETRIX_FILEPATH = DATA_DIR / "Axiom_PMRA.na35.annot.csv"


def get_watson_crick_complement(base: str) -> str:
    """Calculate Watson-Crick Complement."""
    wc_dict = {"A": "T", "T": "A", "G": "C", "C": "G"}
    return wc_dict[base]


def liftover_38_from_37(variant: str) -> str | None:
    """Liftover a variant to genome build 38 from 37."""
    chrom, pos, ref, alt = variant.split(":")
    # liftover doesn't deal gracefully with MT variants, but we don't need them anyway
    if "MT" in chrom:
        return None
    locations = converter[chrom][int(pos)]
    if locations is None or len(locations) != 1:
        logger.info("Variant %s had a non-unique location, couldn't lift over", variant)
        return None
    chrom38, pos38, _strand38 = locations[0]
    variant38 = ":".join([chrom38, str(pos38), ref, alt])
    assert_true("lifted-over variant starts with 'chr'", variant38.startswith("chr"))
    return variant38


def _load_illumina_df() -> pd.DataFrame:
    comment_rows = 7
    columns_to_keep = ["Chr", "MapInfo", "SNP", "RefStrand"]
    illumina_df = pd.read_csv(ILLUMINA_FILEPATH, skiprows=comment_rows)
    assert_equals(
        "Set of genome builds",
        {37.1},
        "actual set of genome builds",
        set(illumina_df.GenomeBuild.dropna()),
    )
    return illumina_df[columns_to_keep]


def load_illumina_callset() -> pd.Series:
    """Load list of variants for illumina chip."""
    illumina_df = _load_illumina_df()
    illumina_variants = _process_illumina_df(illumina_df)
    assert_equals(
        "number of illumina variants", 578822, "recovered number of variants", len(illumina_variants)
    )
    return illumina_variants


def _process_illumina_df(illumina_df: pd.DataFrame) -> pd.Series:
    variants = []
    liftover_attempts = 0
    liftover_failures = 0
    for _i, row in tqdm.tqdm(illumina_df.iterrows(), total=len(illumina_df)):
        chromosome = str(row.Chr)
        if chromosome in ["X", "Y", "0", "MT", "XY"]:
            continue
        try:
            position = str(int(row.MapInfo))
        except ValueError:
            continue
        position = str(int(row.MapInfo))
        liftover_attempts += 1
        results = converter[int(chromosome)][int(position)]
        if len(results) == 1:
            chromosome38, position38, strand38 = results[0]
            position38 = str(position38)
        else:
            logger.debug("couldn't liftover: %s %s", chromosome, position)
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
    logger.info("liftover failure rate: %1.2f%%", liftover_failure_rate_pct)
    return pd.Series(variants)


def _load_affymetrix_df() -> pd.DataFrame:
    affymetrix_df = pd.read_csv(AFFYMETRIX_FILEPATH, comment="#", index_col=0, dtype={"Chromosome": str})
    columns_to_keep = ["Chromosome", "Physical Position", "Ref Allele", "Alt Allele"]
    assert_equals("positive strand", {"+"}, "actual set of strands", set(affymetrix_df.Strand))
    assert_equals(
        "differences between Physical Position and Position End",
        {0},
        "actual differences",
        set(affymetrix_df["Physical Position"] - affymetrix_df["Position End"]),
    )

    return affymetrix_df[columns_to_keep]


def _process_affymetrix_df(affymetrix_df: pd.DataFrame) -> pd.Series:
    """Load list of variants for Affymetrix chip."""
    variants38 = []
    for _i, row in affymetrix_df.iterrows():
        variant = ":".join(
            [
                "chr" + (row.Chromosome),
                str(row["Physical Position"]),
                row["Ref Allele"],
                row["Alt Allele"],
            ]
        )
        variant38 = liftover_38_from_37(variant)
        variants38.append(variant38)
    return pd.Series(variants38, index=affymetrix_df.index).dropna()


def load_affymetrix_callset() -> pd.Series:
    return _process_affymetrix_df(_load_affymetrix_df())


def calculate_shared_illumina_affymetrix_variants() -> pd.DataFrame:
    """Calculate intersection of illumina, affymetrix variants and write result to disk."""
    illumina_variants = load_illumina_callset()
    affy_variants = load_affymetrix_callset()
    shared_variants = pd.DataFrame(sorted(set(illumina_variants).intersection(affy_variants)))
    shared_variants.to_csv(
        INTERMEDIATE_DATA_DIR / "shared_illumina_affy_variants.csv", index=False, header=False
    )
    return shared_variants
