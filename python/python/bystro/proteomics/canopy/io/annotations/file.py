from __future__ import annotations

import hashlib
import logging

import pandas as pd

from bystro.proteomics.canopy import Annotations

from . import errors

CONFIG = {
    'skiprows': 8,
    'header': [0],
    'index_col': 0,
}

APPROVED_MD5_SET = set(
    [
        '7d92666369d4e33364b11804f2d1f8ce',  # v4 rev 2 as of 2021-08-27
        '5fa46834ed826eb1e8dba88698cf7a76',  # v4.1 rev 2 as of 2021-08-27
    ]
)


def read_annotations(filepath: str) -> Annotations:
    """Returns an Annotations object from the filepath/name.

    Parameters
    ----------
    filepath: str
        Either the absolute or relative path to the Excel file to be opened.

    Examples
    --------
    >>> annotations = canopy.read_annotations('path/to/annotations.xlsx')

    Returns
    -------
    annotations : Annotations
    """

    # Get md5 checksum of file to know how to read it
    with open(filepath, 'rb') as f:
        readable_hash = hashlib.md5(f.read()).hexdigest()
    if readable_hash not in APPROVED_MD5_SET:
        logging.warning(
            'Unknown annotations file md5. Continuing with provided annotations. Features in this utility may not run as expected.'
        )

    df = pd.read_excel(filepath, engine='openpyxl', dtype=object, **CONFIG)
    annotations = Annotations(data=df.values, index=df.index, columns=df.columns)
    annotations = annotations.dropna(how='all')

    return annotations
