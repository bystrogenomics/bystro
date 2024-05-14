from __future__ import annotations

import re
from copy import deepcopy
from warnings import warn

import pandas as pd

from bystro.proteomics.canopy import Adat
from bystro.proteomics.canopy.data.lift import check_substrings

from . import errors

LIFTING_COLUMN_REGEX = re.compile(
    r'\w+ Scalar v\d+.\d+ (?:\d+K )?to v\d+.\d+(?: \d+K)?'
)


class Annotations(pd.DataFrame):
    """A Pandas `DataFrame` object with additional functionality to help with annotations specific needs.


    Modeled after:
    https://github.com/geopandas

    On subclassing the pandas dataframe:
    https://pandas.pydata.org/pandas-docs/stable/development/extending.html#subclassing-pandas-data-structures
    """

    _metadata = [
        'supported_lifting_matrices',
        'supported_lifting_signal_space',
    ]

    def __init__(self, *args, **kwargs) -> None:
        super(Annotations, self).__init__(*args, **kwargs)
        self._update_supported_lifting_options()

    @property
    def _constructor(self) -> Adat:
        self._update_supported_lifting_options()
        return Annotations

    def __setitem__(self, key, val):
        super().__setitem__(key, val)
        if LIFTING_COLUMN_REGEX.match(key):
            self._update_supported_lifting_options()

    def __delitem__(self, key) -> None:
        super().__delitem__(key)
        if LIFTING_COLUMN_REGEX.match(key):
            self._update_supported_lifting_options()

    def _update_supported_lifting_options(self):
        self.supported_lifting_matrices = set()
        self.supported_lifting_signal_space = set()

        for name in self.columns:
            if LIFTING_COLUMN_REGEX.match(name):
                supported_info = name.split(' ')
                self.supported_lifting_matrices.add(supported_info[0])
                self.supported_lifting_signal_space.add(
                    (supported_info[2], supported_info[5])
                )

    def update_adat_column_meta(self, adat: Adat) -> Adat:
        """Utility to update a provided adat's column metadata to match the annotations object.

        Attempts to update the following column metadata in the adat:
         - SomaId
         - Target
         - TargetFullName
         - UniProt
         - Type
         - Organism
         - EntrezGeneSymbol
         - ExtrezGeneID

        Parameters
        ----------
        adat : Adat
            Canopy Adat object

        Returns
        -------
        updated_adat : Adat
            Canopy Adat object with updated column metadata

        Examples
        --------
        >>> updated_adat = Annotations.update_adat_column_meta(adat)
        """

        xlsx_to_adat_column_map = {
            'SomaId': 'SomaId',
            'Target Name': 'Target',
            'Target Full Name': 'TargetFullName',
            'UniProt ID': 'UniProt',
            'Type': 'Type',
            'Organism': 'Organism',
            'Entrez Gene Name': 'EntrezGeneSymbol',
            'Entrez Gene ID': 'EntrezGeneID',
        }

        seq_ids = self.index.get_level_values('SeqId')
        mod_adat = adat.copy()
        for xlsx_col, adat_col in xlsx_to_adat_column_map.items():
            if adat_col not in adat.columns.names:
                continue
            values_dict = {
                seq_id: col_meta
                for seq_id, col_meta in zip(seq_ids, self[xlsx_col].values)
            }
            mod_adat = mod_adat.replace_keyed_meta(
                axis=1,
                replaced_meta_name=adat_col,
                key_meta_name='SeqId',
                values_dict=values_dict,
            )
        return mod_adat

    def supported_lifting_space_str(self):
        ret_str = ''
        for space in self.supported_lifting_signal_space:
            ret_str += f'from "{space[0]}" to "{space[1]}"'
        return ret_str

    def lift_adat(self, adat: Adat, lift_to_version: str = None) -> Adat:
        """Utility to perform lifting on an adat.

        Parameters
        ----------
        adat : Adat
            Canopy Adat object

        Returns
        -------
        lifted_adat : Adat
            Canopy Adat object with scaled RFU

        Examples
        --------
        >>> lifted_adat = Annotations.lift_adat(adat=adat)
        """
        warn(
            'Annotations.lift_adat() will be deprecated in a future version. Use Adat.lift() instead.'
        )
        self._update_supported_lifting_options()

        # Perform checks to see if this bridging is appropriate for this adat
        adat = adat.copy()
        process_steps = adat.header_metadata['!ProcessSteps']
        if not 'anmlsmp' in process_steps.lower():
            raise errors.AnnotationsLiftingError(
                f'ANML normalized SOMAscan data is required for lifting. Provided norm steps: "{process_steps}"'
            )

        # Get matrix from adat header metadata
        try:
            matrix = adat.header_metadata['StudyMatrix']
        except KeyError:
            matrix = adat.header_metadata['!StudyMatrix']
        if (
            matrix == 'EDTA Plasma'
        ):  # Takes care of the EDTA Plasma --> Plasma conversion so we can look up the column in the annotations df
            matrix = 'Plasma'
        if matrix not in self.supported_lifting_matrices:
            raise errors.AnnotationsLiftingError(
                f'Unsupported matrix: "{matrix}". Supported matrices: {", ".join(self.supported_lifting_matrices)}.'
            )

        # Get assay version from adat header metadata. Prefer SignalSpace (created by lifting apps) if it exists
        if 'SignalSpace' in adat.header_metadata:
            signal_space = adat.header_metadata['SignalSpace']
        else:
            signal_space = adat.header_metadata['!AssayVersion']
        if (
            signal_space.lower() == 'v4'
        ):  # Takes care of the v4 and V4 --> v4.0 conversion so we can look up the column in the annotations df
            signal_space = 'v4.0'

        # Check to see if we can perform this lifting with the assay version(s) provided
        if lift_to_version:
            if (
                signal_space,
                lift_to_version,
            ) not in self.supported_lifting_signal_space:
                raise errors.AnnotationsLiftingError(
                    f'Unsupported lifting from "{signal_space}" to "{lift_to_version}". Supported lifting: {self.supported_lifting_space_str()}.'
                )
        else:
            possible_lifts = [
                lift
                for lift in self.supported_lifting_signal_space
                if lift[0] == signal_space
            ]
            if not possible_lifts:
                raise errors.AnnotationsLiftingError(
                    f'Unsupported lifting from: "{signal_space}". Supported lifting: {self.supported_lifting_space_str()}.'
                )
            elif len(possible_lifts) > 1:
                raise errors.AnnotationsLiftingError(
                    f'Too many lifting options. Please provide a value for the argument "lift_to_version". Supported lifting: {self.supported_lifting_space_str()}.'
                )
            else:
                lift_to_version = possible_lifts[0][1]

        # Build column name & get scalars
        an_lifting_column = None
        for col in self.columns:
            tests = (
                matrix in col,
                "Scalar" in col,
                check_substrings(col, signal_space, lift_to_version),
            )
            if all(tests):
                an_lifting_column = col
        if not an_lifting_column:
            an_lifting_column = f'{matrix} Scalar {signal_space} to {lift_to_version}'
        scalars = self[an_lifting_column].copy().fillna(1.0)
        # I don't want to modify the annotations in case the object is used elsewhere.
        scalars.index = self['SeqId']

        # Check if seq ids will broadcast between adat & annotations (symmetric difference)
        sym_diff = set(scalars.index) ^ set(adat.columns.get_level_values('SeqId'))
        if sym_diff:
            raise errors.AnnotationsLiftingError(
                'Unable to perform lifting due to analyte mismatch between adat & annotations. Has either file been modified?'
            )

        # Scale adat
        scaled_adat = adat.multiply(scalars, axis='columns', level='SeqId').round(1)
        scaled_adat.header_metadata = deepcopy(adat.header_metadata)
        scaled_adat.header_metadata[
            '!ProcessSteps'
        ] += f', Lifting Bridge ({signal_space} -> {lift_to_version})'
        scaled_adat.header_metadata['SignalSpace'] = lift_to_version

        return scaled_adat
