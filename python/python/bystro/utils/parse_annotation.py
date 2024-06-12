import csv
import gzip
import sys


def parse_tsv_streaming(input_file_path, output_file_path):
    """
    This function takes a Bystro annotation TSV, and explodes the refSeq fields
    on the refSeq.name field, so that each row has a single refSeq transcript, 
    and 1 set of refSeq annotations relative to that transcript
    """
    is_input_gzipped = input_file_path.endswith(".gz")
    is_output_gzipped = output_file_path.endswith(".gz")

    # Open the input file
    if is_input_gzipped:
        infile = gzip.open(input_file_path, "rt", encoding="utf-8")  # noqa: SIM115
    else:
        infile = open(input_file_path, "r", encoding="utf-8")  # noqa: SIM115

    if is_output_gzipped:
        outfile = gzip.open(output_file_path, "wt", newline="", encoding="utf-8")  # noqa: SIM115
    else:
        outfile = open(output_file_path, "w", newline="", encoding="utf-8")  # noqa: SIM115

    reader = csv.DictReader(infile, delimiter="\t")
    fieldnames = reader.fieldnames

    if fieldnames is None:
        raise ValueError("No fieldnames found in the input file")

    # Identify refseq fields
    refseq_fields = [field for field in fieldnames if field.startswith("refSeq.")]
    primary_key = "refSeq.name"

    assert primary_key in refseq_fields

    non_refseq_fields = [field for field in fieldnames if field not in refseq_fields]

    writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()

    for row in reader:
        # For refSeq fields, we annotate one entry per disrupted base in indels
        # We deduplicate the output, so we just need to figure out the maximum length
        exploded_rows = []
        refseq_values = {field: row[field].split("|") for field in refseq_fields}
        max_length = max(len(values) for values in refseq_values.values())

        for field in refseq_fields:
            # Because we deduplicate the output at every delimiter level / array nesting level
            # We simply need to re-expand deduplicated fields to the maximum length
            # Of course if the length of all fields is 1, we don't need to do anything
            if len(refseq_values[field]) < max_length:
                assert len(refseq_values[field]) == 1
                refseq_values[field] = (
                    refseq_values[field] * (max_length // len(refseq_values[field]))
                    + refseq_values[field][: max_length % len(refseq_values[field])]
                )

        new_row = {field: row[field] for field in non_refseq_fields}
        for i in range(max_length):
            # Within each indel position record, we may have multiple refSeq entries, delimited by ";"
            refseq_field_values = {
                field: refseq_values[field][i].split(";") if refseq_values[field][i] else []
                for field in refseq_fields
            }

            # Records are always output relative to the primary key (refSeq.name)
            # but since we deduplicate every field when all values at this delimiter level are identical
            # we need to check what the lenght of the primary key is
            # If it's 1, we just output the row as is, because we cannot get to a lower level of detail
            # relative to the transcript
            # Else we output one transcript per row
            # Note that when the primary key has a single value, other fields may have multiple values
            # And when the primary key has multiple values they are always in the length of the primary
            # key
            primary_key_val = refseq_values[primary_key][i].split(";")

            if len(primary_key_val) == 1:
                new_row2 = new_row.copy()

                for key in refseq_fields:
                    new_row2[key] = refseq_values[key][i]

                exploded_rows.append(new_row2)
                continue

            for y in range(len(primary_key_val)):
                new_row2 = new_row.copy()

                for field, value in refseq_field_values.items():
                    if len(value) == 1:
                        new_row2[field] = value[0]
                    else:
                        new_row2[field] = value[y]

                exploded_rows.append(new_row2)

        for exploded_row in exploded_rows:
            writer.writerow(exploded_row)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python explode_genes.py <annotation_file_path> <output_file_path>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    parse_tsv_streaming(input_file_path, output_file_path)
