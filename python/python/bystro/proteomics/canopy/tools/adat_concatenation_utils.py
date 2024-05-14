from __future__ import annotations
from typing import List, Dict
from ..adat import Adat
from copy import deepcopy
import re
import warnings


def convert_somamer_metadata_to_source(adats: List[Adat], source_adat: Adat) -> List[Adat]:
    """Given a list of adat dataframes and a 'source' adat dataframe, returns the list with updated somamer metadata.

    An adat concatenation method that prepares adats with disparate somamer metadata for merging by unifying their
    somamer column metadata. Since the concatenator requires select column metadata fields to match, this is required
    if there has been a change in protein effective date across adats.

    Parameters
    ----------
    adats : List[Adat]
        List of Adat objects

    source_adat : Adat
        An Adat object

    Returns
    -------
    modified_adats : List[Adat]
        A list of Adat objects whose somamer column metadata has been modified to match the source_adat's.

    Examples
    --------
    >>> adat = convert_somamer_metadata_to_source([adat1, adat2, adat3], adat1)
    >>> adat = convert_somamer_metadata_to_source([adat1, adat2, adat3], adat4)
    """
    new_meta_adats = []
    for adat in adats:
        new_meta_adat = adat.update_somamer_metadata_from_adat(source_adat)
        new_meta_adats.append(new_meta_adat)

    return new_meta_adats


def _merge_subheader(master_subheader_list: Dict, to_be_merged_subheader_list: Dict) -> Dict:
    for to_be_merged_entry in to_be_merged_subheader_list:
        merged = False
        for master_value_entry in master_subheader_list:
            if to_be_merged_entry['value'] == master_value_entry['value']:
                master_value_entry['adat_ids'] += to_be_merged_entry['adat_ids']
                merged = True
                break
        if not merged:
            master_subheader_list.append(to_be_merged_entry)

        return master_subheader_list


def _merge_headers(all_header_metadata: List[Dict]) -> Dict:
    merged_header_metadata = {}
    for header_metadata in all_header_metadata:
        for subheader, values_array in header_metadata.items():
            if subheader not in merged_header_metadata:
                merged_header_metadata[subheader] = values_array
            else:
                merged_header_metadata[subheader] = _merge_subheader(deepcopy(merged_header_metadata[subheader]), values_array)
    return merged_header_metadata


def _get_adat_ids(adat: Adat) -> List[str]:
    adat_ids = []
    title_subheader = adat.header_metadata['!Title']

    # Check to see if this has already been concatenated and return all the adat_ids
    if type(title_subheader) == list and 'adat_ids' in title_subheader[0]:
        for title_info in title_subheader:
            adat_ids += title_info['adat_ids']
        return adat_ids

    # Otherwise, build the adat ids
    plate_ids = adat.index.get_level_values('PlateId')
    plate_ids = list(dict.fromkeys(plate_ids))  # Gets unique plate ids & maintains order
    # See about making something unique and non-redundant
    for plate_id in plate_ids:
        if title_subheader in plate_id:
            adat_ids.append(plate_id)
        else:
            adat_ids.append(title_subheader + '_' + plate_id)
    return adat_ids


def _convert_header_meta_to_contain_adat_id(header_metadata: Dict, adat_ids: List[str]) -> Dict:
    for subheader, value in header_metadata.items():
        # See if the subheader has already been formatted by a previous concatenation, otherwise convert it
        if type(header_metadata[subheader]) == list and 'adat_ids' in header_metadata[subheader][0]:
            continue
        else:
            header_metadata[subheader] = [{
                "adat_ids": adat_ids,
                "value": value
            }]
    return header_metadata


def _get_all_header_metadata_by_plate(adats: List[Adat]) -> List[Dict]:
    all_header_metadata = []
    for adat in adats:
        adat_ids = _get_adat_ids(adat)
        converted_header_metadata = _convert_header_meta_to_contain_adat_id(adat.header_metadata, adat_ids)
        all_header_metadata.append(converted_header_metadata)
    return all_header_metadata


def _simplify_header_metadata(header_metadata_with_adat_ids: Dict) -> Dict:
    simple_header_metadata = {}
    for subheader, items in header_metadata_with_adat_ids.items():
        if len(items) == 1 and subheader != '!Title':
            simple_header_metadata[subheader] = items[0]['value']
        else:
            simple_header_metadata[subheader] = items
    return simple_header_metadata


def robust_merge_adat_headers(adats: List[Adat]) -> List[Adat]:
    """Given a list of adat dataframes, returns the list with merged & matching header metadata.

    An adat concatenation method that prepares adats with disparate header metadata for merging by unifying their
    header metadata. Since the concatenator, by default, has specific strategies for each portion of the header,
    this function unifies all headers to contain all necessary information with zero data loss where the concatenator
    can simply require all fields across the adats to be an exact match.

    Parameters
    ----------
    adats : List[Adat]
        List of Adat objects

    Returns
    -------
    modified_adats : List[Adat]
        A list of Adat objects whose header column metadata has been modified to contain all data across adats & match.

    Examples
    --------
    >>> adat = robust_merge_adat_headers([adat1, adat2, adat3])
    >>> adat = robust_merge_adat_headers([adat1, adat2, adat3])
    """
    all_header_metadata = _get_all_header_metadata_by_plate(adats)

    adat_id_merged_header = _merge_headers(all_header_metadata)
    merged_header = _simplify_header_metadata(adat_id_merged_header)

    new_adats = []
    for adat in adats:
        new_adat = adat.copy()
        new_adat.header_metadata = merged_header
        new_adats.append(new_adat)

    return new_adats


def order_merge_row_meta_names(all_row_meta_names: List[List[str]]) -> List[str]:
    typical_row_meta_order = [
        'PlateId',
        'PlateRunDate',
        'ScannerID',
        'PlatePosition',
        'SlideId',
        'Subarray',
        'SampleId',
        'SampleType',
        'PercentDilution',
        'SampleMatrix',
        'Barcode',
        'Barcode2d',
        'SampleName',
        'SampleNotes',
        'AliquotingNotes',
        'SampleDescription',
        'AssayNotes',
        'TimePoint',
        'ExtIdentifier',
        'SsfExtId',
        'SampleGroup',
        'SiteId',
        'TubeUniqueID',
        'CLI'
    ]

    norm_regex_name_match = [
        r'HybControlNormScale',
        r'RowCheck',
        r'NormScale_.+',
        r'ANMLFractionUsed_.+',
    ]
    combined_regex = '(' + ')|('.join(norm_regex_name_match) + ')'

    all_meta_across_adats = []
    seen_meta = set()
    # Compile all metadata across adats
    for row_meta_names in all_row_meta_names:
        for name in row_meta_names:
            if name not in seen_meta:
                all_meta_across_adats.append(name)
                seen_meta.add(name)

    # Create list that will be the final order
    ordered_metadata_names = []

    # Add to list the names that appear in the typical names
    ordered_metadata_names += [name for name in typical_row_meta_order if name in all_meta_across_adats]

    # Add to the list the names that have not already been added and are not normalization names
    for name in all_meta_across_adats:
        if name not in ordered_metadata_names and not re.match(combined_regex, name):
            ordered_metadata_names.append(name)

    # Add the normalization names in the order of the regex
    for re_str in norm_regex_name_match:
        for name in all_meta_across_adats:
            if re.match(re_str, name) and name not in ordered_metadata_names:
                ordered_metadata_names.append(name)

    return ordered_metadata_names


def unify_row_meta_column_names(adats: List[Adat]) -> List[Adat]:
    """Given a list of adat dataframes, returns the list of adats with matching row metadata columns.

    An adat concatenation method that prepares adats with disparate row metadata for merging by unifying their
    row metadata. Since the core concatenator, by default, requires that the row metadata columns match exactly,
    this function unifies all columns to contain all names across adats.

    Parameters
    ----------
    adats : List[Adat]
        List of Adat objects

    Returns
    -------
    modified_adats : List[Adat]
        A list of Adat objects whose header column metadata has been modified to contain all data across adats & match.

    Examples
    --------
    >>> modified_adats = coalesce_row_meta_column_names(adats)
    """
    all_row_meta_names = []
    for adat in adats:
        all_row_meta_names.append(list(adat.index.names))
    ordered_row_meta_names = order_merge_row_meta_names(all_row_meta_names)

    new_adats = []
    for adat in adats:
        names_missing_in_adat = [name for name in ordered_row_meta_names if name not in adat.index.names]
        for name in names_missing_in_adat:
            warnings.warn(f'Adding column to adat: {name}')
            adat[name] = ['' for i in range(adat.shape[0])]
        new_adat = adat.reset_index().set_index(ordered_row_meta_names)
        new_adats.append(new_adat)
    return new_adats


def prepare_rfu_matrix_for_inner_merge(adats: List[Adat]) -> List[Adat]:
    # Get the subset of seqIds across adats
    seq_id_subset = set(adats[0].columns.get_level_values('SeqId'))
    for adat in adats[1:]:
        current_seq_ids = set(adat.columns.get_level_values('SeqId'))
        seq_id_subset = seq_id_subset.intersection(current_seq_ids)

    # Print out the seq ids removed in each adat
    for adat in adats:
        current_seq_ids = set(adat.columns.get_level_values('SeqId'))
        symmetric_difference = seq_id_subset.symmetric_difference(current_seq_ids)
        removed_seq_ids = ', '.join(symmetric_difference)
        plate_ids = ', '.join(set(adat.index.get_level_values('PlateId')))
        if removed_seq_ids:
            warnings.warn(f'Removing seqIds from {plate_ids}: {removed_seq_ids}')

    # Remove the seq ids from the adats:
    seq_id_subset = tuple(seq_id_subset)
    new_adats = []
    for adat in adats:
        current_seq_ids = adat.columns.get_level_values('SeqId')
        keep_drop = [True if seq_id in seq_id_subset else False for seq_id in current_seq_ids]
        new_adats.append(adat.iloc[:, keep_drop].copy())

    return new_adats
