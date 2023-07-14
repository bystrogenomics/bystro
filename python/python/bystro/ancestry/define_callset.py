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


def get_watson_crick_complement(base: str) -> str:
    """Calculate Watson-Crick Complement."""
    wc_dict = {"A": "T", "T": "A", "G": "C", "C": "G"}
    return wc_dict[base]


def liftover_38_from_37(variant: str) -> str | None:
    """Liftover a variant to genome build 38 from 37."""
    chrom, pos, ref, alt = variant.split(":")
    locations = converter[int(chrom.lstrip("chr"))][int(pos)]
    if locations is None or len(locations) != 1:
        logger.info("Variant %s had a non-unique location, couldn't lift over", variant)
        return None
    chrom38, pos38, _strand38 = locations[0]
    variant38 = ":".join([chrom38, str(pos), ref, alt])
    assert_true("lifted-over variant starts with 'chr'", variant38.startswith("chr"))
    return variant38


def load_illumina_callset() -> pd.Series:
    """Load list of variants for illumina chip."""
    illumina_df = pd.read_csv(ILLUMINA_FILEPATH, skiprows=7)
    assert_equals(
        "Set of genome builds",
        {37.1},
        "actual set of genome builds",
        set(illumina_df.GenomeBuild.dropna()),
    )
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
    assert_equals("number of illumina variants", 578822, "recovered number of variants", len(variants))
    return pd.Series(variants)


def load_affymetrix_callset() -> pd.Series:
    """Load list of variants for Affymetrix chip."""
    chip_df = pd.read_csv(
        DATA_DIR / "Axiom_PMRA.na35.annot.csv",
        comment="#",
        index_col=0,
    )
    assert_equals("positive strand", {"+"}, "actual set of strands", set(chip_df.Strand))
    assert_equals(
        "differences between Physical Position and Position End",
        {0},
        "actual differences",
        set(chip_df["Physical Position"] - chip_df["Position End"]),
    )
    variants = []
    for _i, row in chip_df.iterrows():
        variant = ":".join(
            [
                "chr" + str(row.Chromosome),
                str(row["Physical Position"]),
                row["Ref Allele"],
                row["Alt Allele"],
            ]
        )

        variants.append(variant)
    variants38 = [liftover_38_from_37(v) for v in variants]
    chip_df["variant"] = variants38
    return chip_df.variant


def calculate_shared_illumina_affymetrix_variants() -> pd.DataFrame:
    """Calculate intersection of illumina, affymetrix variants and write result to disk."""
    illumina_variants = load_illumina_callset()
    affy_variants = load_affymetrix_callset()
    shared_variants = pd.DataFrame(sorted(set(illumina_variants).intersection(affy_variants)))
    shared_variants.to_csv(
        INTERMEDIATE_DATA_DIR / "shared_illumina_affy_variants.csv", index=False, header=False
    )
    return shared_variants
