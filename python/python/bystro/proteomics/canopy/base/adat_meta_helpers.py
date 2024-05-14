from __future__ import annotations
from bystro.proteomics.canopy.errors import AdatKeyError, AdatMetaError
from typing import Union, List, Set, Tuple, Dict
from bystro.proteomics.canopy.tools.pandas import get_pd_axis
import numpy as np
import warnings
import pandas as pd


class AdatMetaHelpers:
    """A collection of methods to help with altering the adat metadata and the adat based on the metadata.
    """
    def _filter_on_meta(self, axis: int, name: str, values: Union[List(str), Set(str), Tuple(str)], include: bool = True) -> Adat:

        # Check to see if values is the right variable type
        if not isinstance(values, (list, tuple, set)):
            raise TypeError('"values" must be a list, tuple, or set.')
        else:
            values = set(values)

        # Setup array and get appropriate multiindex
        keep = []
        metadata = get_pd_axis(self, axis)

        # Check to ensure all values are in the metadata
        metadata_values = set(metadata.get_level_values(name))
        if not metadata_values.issuperset(values):
            raise KeyError(f'Some or all provided values not found in metadata column, {name}.')

        # Iterate over the selected multiindex and fill the keep array
        for value in metadata.get_level_values(name):
            if value in values:
                keep.append(True)
            else:
                keep.append(False)

        # Invert the keep array if we are excluding the chosen values
        if not include:
            keep = np.invert(keep)

        # Subset the adat
        if axis == 0:
            adat = self.loc[keep]
        elif axis == 1:
            adat = self.loc[:, keep]

        # Return the subsetted adat (default) or modify the current adat in place
        return adat.copy()

    def _filter_meta(self, axis: int, names: Union[List(str), Set(str), Tuple(str)], include: bool) -> Adat:

        # Check to see if names is the right variable type
        if not isinstance(names, (list, tuple, set)):
            raise TypeError('"values" must be a list, tuple, or set.')
        else:
            names = set(names)

        # Make a copy of the df (what we will eventually return) & grab the multiindex
        adat = self.copy()
        metadata = get_pd_axis(adat, axis)

        # Double check to make sure names exist in multiindex
        for name in names:
            if name not in metadata.names:
                raise AdatKeyError(f'Name, "{name}", not found in multiindex')

        # Filter down the metadata
        for name in metadata.names:
            if name not in names and include:
                metadata = metadata.droplevel(name)
            if name in names and not include:
                metadata = metadata.droplevel(name)

        # Assign the metadata to the appropriate place
        if axis == 0:
            adat.index = metadata
        else:
            adat.columns = metadata

        return adat

    def _insert_meta(self, axis: int, name: str, values: Union[List(str), Tuple(str)], replace: bool) -> Adat:

        adat = self.copy()
        if axis == 0:
            if not replace and name in adat.index.names:
                raise AdatKeyError('Name already exists in index, use `adat.replace_meta` instead.')
            elif replace and name not in adat.index.names:
                raise AdatKeyError('Name does not exists in index, use `adat.insert_meta` instead.')
            index_df = adat.index.to_frame()
            index_df[name] = values
            adat.index = pd.MultiIndex.from_frame(index_df)

        elif axis == 1:
            if not replace and name in adat.columns.names:
                raise AdatKeyError('Name already exists in columns, use `adat.replace_meta` instead.')
            elif replace and name not in adat.columns.names:
                raise AdatKeyError('Name does not exists in columns, use `adat.insert_meta` instead.')

            columns_df = adat.columns.to_frame()
            columns_df.loc[:, name] = values
            adat.columns = pd.MultiIndex.from_frame(columns_df)

        return adat

    def exclude_on_meta(self, axis: int, name: str, values: Union[List(str), Set(str), Tuple(str)]) -> Adat:
        """Returns an adat with rfu rows or columns excluded given the multiindex name and values to exclude on.

        Parameters
        ----------
        axis : int
            The metadata/multiindex to operate on:
            0 - row metadata,
            1 - column metadata

        name : str
            The name of the metadata/multiindex row/column to filter based on.

        values : List(str) | Set(str) | Tuple(str)
            The values to filter on.  Can be a tuple, list, or set.

        Returns
        -------
        adat : Adat

        Examples
        --------
        >>> new_adat = adat.exclude_on_meta(axis=0, name='Barcode', values=['00001'])
        >>> new_adat = adat.exclude_on_meta(axis=1, name='SeqId', values=['10000-01', '12345-10'])
        >>> new_adat = adat.exclude_on_meta(axis=1, name='Type', values=['Spuriomer'])
        """
        return self._filter_on_meta(axis, name, values, include=False)

    def pick_on_meta(self, axis: int, name: str, values: Union[List(str), Set(str), Tuple(str)]) -> Adat:
        """Returns an adat with rfu rows or columns excluded given the multiindex name and values to keep.

        Parameters
        ----------
        axis : int
            The metadata/multiindex to operate on:
            0 - row metadata,
            1 - column metadata

        name : str
            The name of the metadata/multiindex row/column to filter based on.

        values : List(str) | Set(str) | Tuple(str)
            The values to filter on.  Can be a tuple, list, or set.

        Returns
        -------
        adat : Adat

        Examples
        --------
        >>> new_adat = adat.pick_on_meta(axis=0, name='Barcode', values=['00001'])
        >>> new_adat = adat.pick_on_meta(axis=1, name='SeqId', values=['10000-01', '12345-10'])
        >>> new_adat = adat.pick_on_meta(axis=1, name='Type', values=['Spuriomer'])
        """

        return self._filter_on_meta(axis, name, values, include=True)

    def pick_meta(self, axis: int, names: Union[List(str), Set(str), Tuple(str)]) -> Adat:
        """Returns an adat with excluded metadata/multiindices given the names to keep.

        Parameters
        ----------
        axis : int
            The metadata/multiindex to operate on:
            0 - row metadata,
            1 - column metadata

        names : List(str) | Set(str) | Tuple(str)
            The names to filter on.  Can be a tuple, list, or set.

        Returns
        -------
        adat : Adat

        Examples
        --------
        >>> new_adat = adat.pick_meta(axis=0, names=['Barcode'])
        >>> new_adat = adat.pick_meta(axis=1, names=['SeqId'])
        """
        return self._filter_meta(axis, names, include=True)

    def exclude_meta(self, axis: int, names: Union[List(str), Set(str), Tuple(str)]) -> Adat:
        """Returns an adat with excluded metadata/multiindices given the names to exclude.

        Parameters
        ----------
        axis : int
            The metadata/multiindex to operate on:
            0 - row metadata,
            1 - column metadata

        names : List(str) | Set(str) | Tuple(str)
            The names to filter on.  Can be a tuple, list, or set.

        Returns
        -------
        adat : Adat

        Examples
        --------
        >>> new_adat = adat.exclude_meta(axis=0, names=['Barcode'])
        >>> new_adat = adat.exclude_meta(axis=1, names=['SeqId'])
        """

        return self._filter_meta(axis, names, include=False)

    def insert_meta(self, axis: int, name: str, values: Union[List(str), Tuple(str)]) -> Adat:
        """Returns an adat with the given metadata/multiindices added.

        Metadata/multiindex name must not already exist in the adat.

        Parameters
        ----------
        axis : int
            The metadata/multiindex to operate on:
            0 - row metadata,
            1 - column metadata

        name : str
            The name of the index to be added.

        values : List(str) | Tuple(str)
            Values to be added to the metadata/multiindex.  Can be a tuple or list

        Returns
        -------
        adat : Adat

        Examples
        --------
        >>> new_adat = adat.insert_meta(axis=0, name='NewBarcode', values=[1, 2, 3, 4])
        >>> new_adat = adat.insert_meta(axis=1, name='NewType', values=['Protein', 'Protein'])
        """
        return self._insert_meta(axis, name, values, replace=False)

    def replace_meta(self, axis: int, name: str, values: Union[List(str), Tuple(str)]) -> Adat:
        """Returns an adat with the given metadata/multiindices added.

        Metadata/multiindex must already exist in the adat.

        Parameters
        ----------
        axis : int
            The metadata/multiindex to operate on:
            0 - row metadata,
            1 - column metadata

        name : str
            The name of the index to be added.

        values : List(str) | Tuple(str)
            Values to be added to the metadata/multiindex.  Can be a tuple or list

        Returns
        -------
        adat : Adat

        Examples
        --------
        >>> new_adat = adat.replace_meta(axis=0, name='Barcode', values=[1, 2, 3, 4])
        >>> new_adat = adat.replace_meta(axis=1, name='Type', values=['Protein', 'Protein'])
        """
        return self._insert_meta(axis, name, values, replace=True)

    def insert_keyed_meta(self, axis: int, inserted_meta_name: str, key_meta_name: str, values_dict: Dict(str, str)) -> Adat:
        """Inserts metadata into Adat given a dictionary of values keyed to existing metadata.

        If a key does not exist in values_dict, the function will fill in missing data with empty strings
        values and create a warning to notify the user.

        Parameters
        ----------
        axis : int
            The metadata/multiindex to operate on:
            0 - row metadata,
            1 - column metadata

        inserted_meta_name : str
            The name of the index to be added.

        key_meta_name : str
            The name of the index to use as the key-map.

        values_dict : Dict(str, str)
            Values to be added to the metadata/multiindex keyed to the existing values in `key_meta_name`.

        Returns
        -------
        adat : Adat

        Examples
        --------
        >>> new_adat = adat.insert_keyed_meta(axis=0, inserted_meta_name='NewBarcode', key_meta_name='Barcode', values_dict={"J12345": "1"})
        >>> new_adat = adat.insert_keyed_meta(axis=1, inserted_meta_name='NewProteinType', key_meta_name='Type', values_dict={"Protein": "Buffer")
        """

        values = []
        metadata = get_pd_axis(self, axis)
        key_meta = metadata.get_level_values(key_meta_name)

        if inserted_meta_name in metadata.names:
            raise AdatKeyError('Name already exists in index, use `adat.replace_keyed_meta` instead.')

        for key in key_meta:
            if key in values_dict:
                values.append(values_dict[key])
            else:
                values.append('')

        if None in values:
            warnings.warn('Empty string values inserted into metadata.', category=Warning)

        return self.insert_meta(axis, inserted_meta_name, values)

    def replace_keyed_meta(self, axis: int, replaced_meta_name: str, values_dict: Dict(str, str), key_meta_name: str = None) -> Adat:
        """Updates metadata in an Adat given a dictionary of values keyed to existing metadata.

        If a key does not exist in values_dict, the function will fill in missing data with pre-existing
        values and create a warning to notify the user.

        Parameters
        ----------
        axis : int
            The metadata/multiindex to operate on:
            0 - row metadata,
            1 - column metadata

        replaced_meta_name : str
            The name of the index to be added.

        key_meta_name : str, optional
            The name of the index to use as the key-map.  Will default to `replaced_meta_name` if None.

        values_dict : Dict(str, str)
            Values to be added to the metadata/multiindex keyed to the existing values in `key_meta_name`.

        Returns
        -------
        adat : Adat

        Examples
        --------
        >>> new_adat = adat.replace_keyed_meta(axis=0, inserted_meta_name='Barcode', key_meta_name='SampleType', values_dict={"J12345": "Calibrator"})
        >>> new_adat = adat.replace_keyed_meta(axis=1, inserted_meta_name='Type', key_meta_name='SeqId', values_dict={"12345-6": "ProteinSet1")
        """

        key_meta_name = key_meta_name or replaced_meta_name

        values = []
        metadata = get_pd_axis(self, axis)
        key_meta = metadata.get_level_values(key_meta_name)
        values_to_update = metadata.get_level_values(replaced_meta_name)

        if replaced_meta_name not in metadata.names:
            raise AdatKeyError('Name does not exists in index, use `adat.insert_keyed_meta` instead.')

        warning_str = 'Some keys not provided, using original values for those keys'
        warnings.filterwarnings('once', message=warning_str)
        for key, value in zip(key_meta, values_to_update):
            if key in values_dict:
                values.append(values_dict[key])
            else:
                warnings.warn(warning_str)
                values.append(value)

        return self.replace_meta(axis, replaced_meta_name, values)

    def update_somamer_metadata_from_adat(self, adat: Adat) -> Adat:
        """Given an Adat with different SOMAmer reagent metadata, returns this adat with that somamer metadata.

        An adat method that updates adats with disparate somamer metadata by unifying their somamer column
        metadata. Useful for concatenating since the concatenator requires select column metadata fields to match.
        This is required for concatenation if there has been a change in protein effective date across adats.

        Parameters
        ----------
        adat : Adat
            An Adat object with the metadata you want to use

        Returns
        -------
        modified_adat : Adat
            This Adat whose somamer column metadata has been modified to match the provided adat's.

        Examples
        --------
        >>> new_adat = adat.update_somamer_metadata_from_adat(other_adat)
        >>> new_adat = adat.update_somamer_metadata_from_adat(other_adat)
        """

        # Check to make sure seq_ids & order are identical
        if list(adat.columns.get_level_values('SeqId')) != list(self.columns.get_level_values('SeqId')):
            raise AdatMetaError('SeqIds do not match the provided adat. Unable to perform metadata substitution')

        columns_to_overwrite = [
            'SeqIdVersion', 'SomaId', 'TargetFullName', 'Target', 'UniProt',
            'EntrezGeneID', 'EntrezGeneSymbol', 'Organism', 'Units', 'Type', 'Dilution',
        ]

        new_meta_adat = self.copy()
        # Modify adat for each name in columns_to_overwrite
        for column_name in columns_to_overwrite:

            # Check to see if the column exists. If it doesn't, throw a warning & move on to the next one
            if column_name not in new_meta_adat.columns.names:
                warnings.warn(f'Standard column, {column_name}, not found in column metadata. Continuing to next.')
                continue
            # If it does exist in the source adat but not in the provided adat, we have problems!
            elif column_name not in adat.columns.names:
                AdatMetaError(f'Standard column, {column_name}, not found in provided column metadata but exists in source adat.')

            # Replace metadata
            new_meta_adat = new_meta_adat.replace_meta(axis=1, name=column_name, values=adat.columns.get_level_values(column_name))

        return new_meta_adat

    def reorder_on_metadata(self, axis: int, name: str, source_adat: Adat) -> Adat:
        """Given an Adat with matching metadata in a different order, returns this adat reorganized to match that order.

        An adat method that updates adats with mis-aligned metadata by unifying the order of the columns or rows by the metadata.

        Parameters
        ----------
        axis : int
            The metadata/multiindex to operate on:
            0 - row metadata,
            1 - column metadata

        name : str
            The name of the index to be added.

        source_adat : Adat
            An Adat object with the metadata order you want

        Returns
        -------
        modified_adat : Adat
            This Adat whose rows or columns has been reordered to match the provided adat's.

        Examples
        --------
        >>> new_adat = adat.reorder_on_metadata(axis=1, name='SeqId', other_adat)
        """
        reorder_index = []
        adat = self.copy()
        if axis == 0:
            metadata_order = list(source_adat.index.get_level_values(name))
            for metadata in adat.index.get_level_values(name):
                try:
                    reorder_index.append(metadata_order.index(metadata))
                except ValueError:
                    raise AdatMetaError(f'Source metadata, {metadata}, not found in adat index, {name}')
            adat = adat.iloc[reorder_index]
        elif axis == 1:
            metadata_order = list(source_adat.columns.get_level_values(name))
            for metadata in adat.columns.get_level_values(name):
                try:
                    reorder_index.append(metadata_order.index(metadata))
                except ValueError:
                    raise AdatMetaError(f'Source metadata, {metadata}, not found in adat column, {name}')
            adat = adat.iloc[:, reorder_index]

        return adat
