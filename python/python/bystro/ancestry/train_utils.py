"""Utilities for ancestry model training."""
import re
from collections.abc import Iterable
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")


def get_variant_ids_from_callset(callset: dict[str, Any]) -> np.ndarray:
    """Given a callset generated from scikit.allel, return variant ids in Broad notation."""
    # see https://illumina.github.io/NirvanaDocumentation/core-functionality/variant-ids/
    return np.array(
        [
            "-".join([chrom, str(pos), ref, alt])
            for chrom, pos, ref, alt in zip(
                callset["variants/CHROM"],
                callset["variants/POS"],
                callset["variants/REF"],
                callset["variants/ALT"][:, 0],
                # matrix contains all alts for position-- we only want the first because we're
                # excluding all multi-allelic variants
                strict=True,
            )
        ],
    )


def head(xs: Iterable[T]) -> T:
    """Get first element of xs."""
    return next(iter(xs))


VARIANT_REGEX = re.compile(
    r"""
    ^
    chr(1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22)  # (autosomal) chromosome
    :
    [0-9]+  # position
    :
    [ACGT]  # ref allele
    :
    [ACGT]  # alt allele
    $
    """,
    re.VERBOSE,
)


def is_autosomal_variant(potential_variant: str) -> bool:
    """Determine whether string is a syntactically valid autochromosomal variant."""
    if not VARIANT_REGEX.match(potential_variant):
        return False
    chromosome, position, ref, alt = potential_variant.split(":")
    if ref == alt:
        err_msg = f"Variant {potential_variant} cannot have identical ref and alt alleles"
        raise ValueError(err_msg)
    return True
