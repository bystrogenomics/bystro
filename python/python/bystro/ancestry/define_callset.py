"""Calculate intersection of Illumina and Affymetrix chips."""

import logging
import re

import pandas as pd
import tqdm
from liftover import get_lifter

from bystro.ancestry.asserts import assert_equals, assert_true
from bystro.ancestry.train import DATA_DIR, INTERMEDIATE_DATA_DIR
from bystro.ancestry.train_utils import is_autosomal_variant

logger = logging.getLogger(__name__)

pd.options.future.infer_string = True  # type: ignore

ILLUMINA_FILEPATH = DATA_DIR / "Human660W-Quad_v1_H.csv"
AFFYMETRIX_FILEPATH = DATA_DIR / "Axiom_PMRA.na35.annot.csv"

# TODO: harmonize variant chromosomal coordinates with rsIDs.
pd.options.future.infer_string = True  # type: ignore


def get_watson_crick_complement(base: str) -> str:
    """Calculate Watson-Crick Complement."""
    wc_dict = {"A": "T", "T": "A", "G": "C", "C": "G"}
    return wc_dict[base]


def liftover_38_from_37(variant: str) -> str | None:
    """Liftover a variant to genome build 38 from 37."""
    chrom, pos, ref, alt = variant.split(":")
    # liftover doesn't deal gracefully with MT variants, but we don't need them anyway
    converter = get_lifter("hg19", "hg38")
    locations = converter[chrom][int(pos)]
    if locations is None or len(locations) != 1:
        logger.debug("Variant %s had a non-unique location, couldn't lift over", variant)
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
    return illumina_df[columns_to_keep].dropna()


def load_illumina_variants() -> pd.Series:
    """Load list of variants for illumina Human660W-Quad_v1 chip."""
    illumina_df = _load_illumina_df()
    illumina_variants = _get_variants_from_illumina_df(illumina_df)
    assert_equals(
        "number of autosomal illumina variants after liftover",
        578822,
        "recovered number of variants",
        len(illumina_variants),
    )
    return illumina_variants


def _get_variants_from_illumina_df(illumina_df: pd.DataFrame) -> pd.Series:
    """Extract illumina variants and lift over."""
    variants38 = []
    for _i, row in tqdm.tqdm(illumina_df.iterrows(), total=len(illumina_df)):
        chromosome = str(row.Chr)
        position = str(int(row.MapInfo))
        if match := re.match(r"\[([ACGT])/([ACGT])\]", str(row.SNP)):
            allele1, allele2 = match.groups()
        else:
            continue
        if row.RefStrand == "-":
            allele1 = get_watson_crick_complement(allele1)
            allele2 = get_watson_crick_complement(allele2)
        variant37 = ":".join(["chr" + chromosome, position, allele1, allele2])
        if not is_autosomal_variant(variant37):
            continue
        variants38.append(liftover_38_from_37(variant37))
    illumina_variants = pd.Series(variants38)
    liftover_failure_rate = illumina_variants.isna().mean()
    logger.info("liftover failure rate: %1.2f%%", liftover_failure_rate * 100)
    return illumina_variants.dropna()


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


def _get_variants_from_affymetrix_df(affymetrix_df: pd.DataFrame) -> pd.Series:
    """Extract affymetrix variants and lift over."""
    variants38 = []
    for _i, row in tqdm.tqdm(affymetrix_df.iterrows(), total=len(affymetrix_df)):
        variant = ":".join(
            [
                "chr" + (row.Chromosome),
                str(row["Physical Position"]),
                row["Ref Allele"],
                row["Alt Allele"],
            ]
        )
        if not is_autosomal_variant(variant):
            continue
        variant38 = liftover_38_from_37(variant)
        variants38.append(variant38)
    affymetrix_variants = pd.Series(variants38)
    liftover_failure_rate = affymetrix_variants.isna().mean()
    logger.info("liftover failure rate: %1.2f%%", liftover_failure_rate * 100)
    return affymetrix_variants.dropna()


def load_affymetrix_variants() -> pd.Series:
    """Load list of variants for Affymetrix Axiom PMRA chip."""
    affymetrix_variants = _get_variants_from_affymetrix_df(_load_affymetrix_df())
    assert_equals(
        "number of autosomal affymetrix variants after liftover",
        820710,
        "recovered number of variants",
        len(affymetrix_variants),
    )
    return affymetrix_variants


def calculate_shared_illumina_affymetrix_variants() -> pd.DataFrame:
    """Calculate intersection of illumina, affymetrix variants and write result to disk."""
    illumina_variants = load_illumina_variants()
    affymetrix_variants = load_affymetrix_variants()
    shared_variants = pd.DataFrame(sorted(set(illumina_variants).intersection(affymetrix_variants)))
    assert_equals(
        "number of shared variants", 34319, "number of shared variants obtained", len(shared_variants)
    )
    shared_variants.to_csv(
        INTERMEDIATE_DATA_DIR / "shared_illumina_affy_variants.csv", index=False, header=False
    )
    return shared_variants
