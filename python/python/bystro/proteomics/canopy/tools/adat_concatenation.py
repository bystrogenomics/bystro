from __future__ import annotations
from bystro.proteomics.canopy import Adat
from bystro.proteomics.canopy.tools.errors import AdatConcatError
from typing import List, Dict
from . import adat_concatenation_utils
import copy
import re


def _set_addition(key, value1, value2):
    value_set_1 = set([x.strip() for x in value1.split(',')])
    value_set_2 = set([x.strip() for x in value2.split(',')])
    value_set = value_set_1.union(value_set_2)
    str_values = ', '.join(sorted(value_set))
    return str_values


def _exact_match(key, value1, value2):
    if value1 != value2:
        raise AdatConcatError(f'Header metadata mismatch where exact match is required. Key: {key}, Values: {value1}, {value2}')
    return value1


def _append_str_with_pipe(key, value1, value2):
    return value1 + ' | ' + value2


def _null(key, value1, value2):
    return None


def _take_first(key, value1, value2):
    return value1


MERGE_ACTIONS = {
    'set_addition': _set_addition,
    'exact_match': _exact_match,
    'append_str_with_pipe': _append_str_with_pipe,
    'null': _null,
    'take_first': _take_first,
}


DEFAULT_MERGE_STRATEGY = {
    'default_action': 'exact_match',
    'properties': {
        'AdatId': 'null',
        '!AdatId': 'null',
        '!AssayRobot': 'set_addition',
        '!CreatedDate': 'set_addition',
        '!EnteredBy': 'set_addition',
        '!ExpDate': 'set_addition',
        'Notes': 'set_addition',
        'RunNotes': 'append_str_with_pipe',
        '!RunNotes': 'append_str_with_pipe',
        'LabLocation': 'set_addition',
        '!Title': 'set_addition',
    },
}


def _concat_header_metadata(adats: List[Adat], merge_strategy=None):
    merge_strategy = merge_strategy or DEFAULT_MERGE_STRATEGY

    # Seed base header with the first adat's header
    base_header = copy.deepcopy(adats[0].header_metadata)

    # Add the rest of the adats' headers
    for adat in adats[1:]:
        for key, value in adat.header_metadata.items():

            # If key is not in the base_header, add it
            if key not in base_header:
                base_header[key] = value

            # If it is in the base header, try to look up a merge strategy & use it
            elif key in merge_strategy['properties']:
                merge_action = MERGE_ACTIONS[merge_strategy['properties'][key]]
                base_header[key] = merge_action(key, base_header[key], value)

            # All else fails, go to the default strategy
            else:
                merge_action = MERGE_ACTIONS[merge_strategy['default_action']]
                base_header[key] = merge_action(key, base_header[key], value)

    return base_header


def _concat_column_metadata(adats: List[Adat]) -> Dict(str, List):
    # Get Col Metadata
    col_metadata = {}
    col_checks = []
    for adat in adats:
        for name in adat.columns.names:
            values = list(adat.columns.get_level_values(name))
            if name == 'ColCheck':
                col_checks.append([True if value == 'PASS' else False for value in values])
                col_metadata['ColCheck'] = []
            elif name in col_metadata:
                if col_metadata[name] != values:
                    raise AdatConcatError('Mismatching column metadata in: ' + name)
            else:
                col_metadata[name] = values

    # Add ColCheck Back if it Exists
    if col_checks:
        for checks in zip(*col_checks):
            if all(checks):
                col_metadata['ColCheck'].append('PASS')
            else:
                col_metadata['ColCheck'].append('FLAG')

    return col_metadata


def _concat_row_metadata(adats: List[Adat]) -> Dict(str, List):
    # Check if Row Metadata Matches
    names = set(adats[0].index.names)
    symmetric_difference = set()
    for adat in adats[1:]:
        symmetric_difference = symmetric_difference.union(names ^ set(adat.index.names))
        names = names.union(symmetric_difference)
    if len(symmetric_difference) > 0:
        raise AdatConcatError('Mismatching index name, ensure row metadata columns match. Names: ' + ', '.join(sorted(symmetric_difference)))

    # Get Row Metadata
    row_metadata = {}
    for adat in adats:
        for name in adat.index.names:
            if name not in row_metadata:
                row_metadata[name] = []
            row_metadata[name] += list(adat.index.get_level_values(name))

    return row_metadata


def _concat_rfus(adats: List[Adat]) -> List[List[float]]:
    # Get RFU Values
    values = []
    for adat in adats:
        values += list(adat.values)
    return values


def concatenate_adats(adats: List[Adat], header_merge_strategy: Dict = None) -> Adat:
    """Given list of compatible adats will return a single adat with all data.

    An adat concatenation method that requires all row and column metadata have the same fields.
    The method also requires select header metadata fields to match (unless overridden via the
    `header_merge_strategy` keyword).

    Parameters
    ----------
    adats : List[Adat]
        List of Adat objects

    header_merge_strategy : Dict (Optional)
        A dictionary containing a 'default_action' and a dictionary of 'properties'.
        The properties dictionary contains key/value pairs of header title and merge
        method.  Will overwrite the existing strategy if provided.

        Available merge methods:

        - 'exact' (default): The fields must be exact in order to allow for the merge

        - 'set_addition': Will split the fields by comma and merge them via set addition (unique values kept)

        - 'append_str_with_pipe': Will merge all fields dilimited by pipes

        - 'null': Will null the field

    Returns
    -------
    adat : Adat
        Concatenated adat

    Examples
    --------
    >>> adat = concatenate_adats([adat1, adat2, adat3])
    >>> adat = concatenate_adats([adat1, adat2, adat3], header_merge_strategy={'default_action': 'null', 'properties': {'AdatId': 'exact'}})
    """

    header_metadata = _concat_header_metadata(adats, merge_strategy=header_merge_strategy)
    column_metadata = _concat_column_metadata(adats)
    row_metadata = _concat_row_metadata(adats)
    rfu_matrix = _concat_rfus(adats)

    adat = Adat.from_features(rfu_matrix, row_metadata, column_metadata, header_metadata)
    return adat


def _quick_concat(adats):
    row_multiindex = adats[0].index
    rfu_matrix = adats[0].values
    for adat in adats[1:]:
        row_multiindex.append(adat.index)
        rfu_matrix += adat.values
    return Adat(
        data=rfu_matrix,
        index=row_multiindex,
        columns=adats[0].columns,
        header_metadata=adats[0].header_metadata
    )


def smart_adat_concatenation(adats, somamer_source_adat=None):
    """Given list of adats and (optionally) a somamer metadata source adat, returns a single adat with all data.

    An smart adat concatenation method that will modify the adats to agree in its row, column, and header metadata.
    Will outer join row metadata, inner join the rfu matrix by seqIds, and merge header metadata so that the
    provenance of the header values is maintained.

    Parameters
    ----------
    adats : List[Adat]
        List of Adat objects

    somamer_source_adat : Adat
        Adat that serves as the source for the SOMAmer Reagent metadata.

    Returns
    -------
    adat : Adat
        Concatenated adat

    Examples
    --------
    >>> adat = smart_adat_concatenation([adat1, adat2, adat3])
    >>> adat = smart_adat_concatenation([adat1, adat2, adat3], somamer_source_adat=adat1)
    """

    # About to change the adats somamer metadata.  Make sure their seqids are the same.
    if type(somamer_source_adat) == Adat:
        adats = adats + [somamer_source_adat]
        
    adats = adat_concatenation_utils.prepare_rfu_matrix_for_inner_merge(adats)

    # Unpack & update if we're updating
    if type(somamer_source_adat) == Adat:
        somamer_source_adat = adats[-1]
        adats = adats[0:-1]
        adats = adat_concatenation_utils.convert_somamer_metadata_to_source(adats, somamer_source_adat)

    header_merge_strategy = {
        'default_action': 'exact_match',
        'properties': {
            'AdatId': 'null',
            '!AdatId': 'null',
        }
    }

    adats = adat_concatenation_utils.robust_merge_adat_headers(adats)
    adats = adat_concatenation_utils.unify_row_meta_column_names(adats)

    concat_adat = concatenate_adats(adats, header_merge_strategy=header_merge_strategy)
    return concat_adat
