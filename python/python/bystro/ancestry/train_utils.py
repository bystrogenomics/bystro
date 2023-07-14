"""Utilities for ancestry model training."""

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
