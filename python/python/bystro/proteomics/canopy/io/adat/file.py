from __future__ import annotations

import csv
import json
import logging
import re
import warnings
from importlib.metadata import version
from typing import Dict, List, TextIO, Tuple, Union

from bystro.proteomics.canopy import Adat
from bystro.proteomics.canopy.io.adat.errors import AdatReadError
from bystro.proteomics.canopy.tools.math import jround


def parse_file(
    f: TextIO,
) -> Tuple[List[List[float]], Dict[str, List[str]], Dict[str, List[str]], Dict[str, str]]:
    """Returns component pieces of an adat given an adat file object.

    Parameters
    ----------
    f : TextIO
        An open adat file object.

    Returns
    -------
    rfu_matrix : List[List[float]]
        An nSample x nSomamer matrix of the RFU data (by row) where each sub-array corresponds to a sample.

    row_metadata : Dict[str, List[str]]
        A dictionary of each column of the row metadata where the key-value
        pairs are column-name and an array of each sample's corresponding metadata

    column_metadata : Dict[str, List[str]]
        A dictionary of each row of the adat column metadata where the key-value pairs are
        row-name and an array of each somamer's corresponding metadata.

    header_metadata : Dict[str, str]
        A dictionary of each row of the header_metadata corresponds to a key-value pair.
    """
    current_section = None

    header_metadata = {}
    column_metadata = {}
    row_metadata = {}
    rfu_matrix = []

    matrix_depth = 0

    reader = csv.reader(f, delimiter="\t")
    for line in reader:
        # Check for trailing Nones
        for index, cell in enumerate(reversed(line)):
            if cell:
                break
            del line[-1]

        # If we see a new section set which portion of the adat we are in & continue to next line
        if "^HEADER" in line[0]:
            current_section = "HEADER"
            continue
        elif "^TABLE_BEGIN" in line[0]:
            current_section = "TABLE"
            continue
        elif "^COL_DATA" in line[0]:
            current_section = "COL_DATA"
            continue
        elif "^ROW_DATA" in line[0]:
            current_section = "ROW_DATA"
            continue

        # Parse the data according to which section of the adat we're reading

        if current_section == "HEADER":
            # Not every key in the header has a value
            if len(line) == 1:
                header_metadata[line[0]] = ""

            # Should be the typical case
            elif len(line) == 2:
                try:
                    header_metadata[line[0]] = json.loads(line[1])
                    if type(header_metadata[line[0]]) != dict:
                        header_metadata[line[0]] = line[1]
                except json.JSONDecodeError:
                    header_metadata[line[0]] = line[1]

            # More than 2 values to a key should never ever happen
            else:
                raise AdatReadError("Unexpected size of header: " + "|".join(line))

            # If we have the report config section, check to see if it was loaded as a dict
            if line[0] == "ReportConfig" and type(header_metadata[line[0]]) != dict:
                warnings.warn(
                    "Malformed ReportConfig section in header.  Setting to an empty dictionary."
                )
                header_metadata[line[0]] = {}

        elif current_section == "COL_DATA":
            # Get the height of the column metadata section & skip the rest of the section
            col_metadata_length = len(line)
            current_section = None

        elif current_section == "ROW_DATA":
            # Get the index of the end of the row metadata section & skip the rest of the section
            row_metadata_offset = len(line) - 1
            current_section = None

        elif current_section == "TABLE":
            # matrix_depth is used to identify if we are in the column
            # metadata section or the row metadata/rfu section
            matrix_depth += 1

            # Column Metadata Section
            if matrix_depth < col_metadata_length:
                column_metadata_name = line[row_metadata_offset]
                column_metadata_data = line[row_metadata_offset + 1 :]

                if column_metadata_name == "SeqId" and re.match(
                    r"\d{3,}-\d{1,3}_\d+", column_metadata_data[0]
                ):
                    warnings.warn(
                        "V3 style seqIds (i.e., 12345-6_7). Converting to V4 Style. The adat file writer has an option to write using the V3 style"
                    )
                    seq_id_data = [x.split("_")[0] for x in column_metadata_data]
                    version_data = [x.split("_")[1] for x in column_metadata_data]
                    column_metadata[column_metadata_name] = seq_id_data
                    column_metadata["SeqIdVersion"] = version_data
                else:
                    column_metadata[column_metadata_name] = column_metadata_data

            # Perform a check to ensure all column metadata is the same length and if not, extend it to the maximum length
            col_meta_lengths = [len(values) for values in column_metadata.values()]
            if len(set(col_meta_lengths)) > 1:
                max_length = max(col_meta_lengths)
                for name, values in column_metadata.items():
                    if len(values) == max_length:
                        continue
                    warnings.warn(f'Adding empty values to column metadata: "{name}"')
                    n_missing_elements = max_length - len(values)
                    append_array = [""] * n_missing_elements
                    new_values = values + append_array
                    column_metadata[name] = new_values

            # Row Metadata Titles
            elif matrix_depth == col_metadata_length:
                row_metadata_names = line[:row_metadata_offset]
                row_metadata = {name: [] for name in row_metadata_names}

            # Row Metadata & RFU Section
            elif matrix_depth > col_metadata_length:
                # Store in row metadata into dictionary
                row_metadata_data = line[:row_metadata_offset]
                for name, data in zip(row_metadata_names, row_metadata_data):
                    row_metadata[name].append(data)

                # Store the RFU data
                rfu_row_data = line[row_metadata_offset + 1 :]
                converted_rfu_row_data = list(map(float, rfu_row_data))
                rfu_matrix.append(converted_rfu_row_data)

    return rfu_matrix, row_metadata, column_metadata, header_metadata


def read_file(filepath: str) -> Adat:
    """DEPRECATED: SEE canopy.read_adat

    WILL BE REMOVED IN A FUTURE RELEASE
    """
    logging.warning(
        "THIS FUNCTION IS DEPRECATED AND WILL BE REMOVED IN A FUTURE RELEASE.\n PLEASE USE `canopy.read_adat` instead."
    )
    return read_adat(filepath)


def read_adat(path_or_buf: Union[str, TextIO]) -> Adat:
    """Returns an Adat from the filepath/name.

    Parameters
    ----------
    path_or_buf : Union[str, TextIO]
        Path or buffer that the file will be read from

    Examples
    --------
    >>> adat = Adat.from_file('path/to/file.adat')

    Returns
    -------
    adat : Adat
    """
    if type(path_or_buf) == str:
        with open(path_or_buf, "r") as f:
            rfu_matrix, row_metadata, column_metadata, header_metadata = parse_file(f)
    else:
        rfu_matrix, row_metadata, column_metadata, header_metadata = parse_file(path_or_buf)

    return Adat.from_features(
        rfu_matrix=rfu_matrix,
        row_metadata=row_metadata,
        column_metadata=column_metadata,
        header_metadata=header_metadata,
    )


def write_file(adat, path: str, round_rfu: bool = True, convert_to_v3_seq_ids: bool = False) -> None:
    """DEPRECATED: SEE canopy.write_adat

    WILL BE REMOVED IN A FUTURE RELEASE
    """
    logging.warning(
        "THIS FUNCTION IS DEPRECATED AND WILL BE REMOVED IN A FUTURE RELEASE.\n PLEASE USE `canopy.write_adat` instead."
    )
    read_adat(adat, path, round_rfu, convert_to_v3_seq_ids)


def write_adat(adat, f: TextIO, round_rfu: bool = True, convert_to_v3_seq_ids: bool = False) -> None:
    """Write this Adat to an adat format data source.

    Parameters
    ----------
    adat : Adat
        Adat Pandas dataframe to be written.

    path : str
        The file path to write to.

    round_rfu : bool
        Rounds the RFU matrix to one decimal place if True,
        otherwise leaves the matrix as-is. (Default = True)

    convert_to_v3_seq_ids : bool
        Combines the column metadata for SeqId and
        SeqIdVersion to the V3 style (12345-6_7)

    Examples
    --------
    >>> pd.write_file(adat, 'path/to/out/filename.adat')
    >>> pd.write_file(adat, 'path/to/out/filename.adat', round_rfu=False)

    Returns
    -------
    None
    """

    # Add version number to header_metadata.  If the field already exists, append to it.
    pkg_version = "Canopy_" + version("canopy")
    if "!GeneratedBy" not in adat.header_metadata:
        adat.header_metadata["!GeneratedBy"] = pkg_version
    elif pkg_version not in adat.header_metadata["!GeneratedBy"]:
        adat.header_metadata["!GeneratedBy"] += ", " + pkg_version

    # Create COL_DATA & ROW_DATA sections
    column_names = adat.columns.names
    column_types = ["String" for name in column_names]

    row_names = adat.index.names
    row_types = ["String" for name in row_names]

    # Start writing the adat using the csv writer
    writer = csv.writer(f, delimiter="\t", lineterminator="\r\n")

    # Checksum must be added with blank value
    writer.writerow(["!Checksum"])

    # Write HEADER section
    writer.writerow(["^HEADER"])
    for row in adat.header_metadata.items():
        # We need to handle the reportconfig in a special way since it has double quotes
        if row[0] == "ReportConfig":
            f.write(row[0] + "\t" + json.dumps(row[1], separators=(",", ":")) + "\r\n")
        else:
            writer.writerow([x for x in row if x is not None])

    # Write COL_DATA section
    writer.writerow(["^COL_DATA"])
    writer.writerow(["!Name"] + column_names)
    writer.writerow(["!Type"] + column_types)

    # Write ROW_DATA section
    writer.writerow(["^ROW_DATA"])
    writer.writerow(["!Name"] + row_names)
    writer.writerow(["!Type"] + row_types)

    # Begin the main section of the adat
    writer.writerow(["^TABLE_BEGIN"])

    # Write the column metadata
    column_offset = [None for i in range(len(row_names))]
    for column_name in column_names:
        # Prep the data
        column_data = adat.columns.get_level_values(column_name)

        # Check if we are converting to the V3 style of adat seqIds
        if column_name == "SeqId" and convert_to_v3_seq_ids:
            version_data = adat.columns.get_level_values("SeqIdVersion")
            column_data = [seq_id + "_" + version for seq_id, version in zip(column_data, version_data)]
        if column_name == "SeqIdVersion" and convert_to_v3_seq_ids:
            continue

        # Create and write the row
        row = []
        row += column_offset
        row += [column_name]
        row += list(column_data)
        writer.writerow(row)

    # Write the row metadata column titles.  Additional tabs added to conform to PX adat structure.
    extra_nones = len(adat.columns.get_level_values(column_names[0])) + 1
    writer.writerow(row_names + [None for x in range(extra_nones)])

    # Write the row metadata and rfu matrix simultaneously
    for i, rfu_row in enumerate(adat.values):
        # Prep the data
        row_metadata = [adat.index.get_level_values(row_name)[i] for row_name in row_names]
        if round_rfu:
            rfu_row = [jround(rfu, 1) for rfu in rfu_row]
        else:
            rfu_row = list(rfu_row)

        # Create and write the row
        row = []
        row += row_metadata
        row += [None]
        row += rfu_row
        writer.writerow(row)
