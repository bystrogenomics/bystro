import json
import os
from contextlib import contextmanager
import subprocess
import tarfile
import gzip

def populate_hash_path_2(row_document: dict, field_path: list, field_value):
    current = row_document
    path_len = len(field_path)

    for k in range(path_len):
        key = field_path[k]

        if key not in row_document:
            current[key] = {}

        if k < path_len - 1:
            current = current[key]
        else:
            current[key] = field_value

    return row_document

def main():
    buffer_size = 48 * 1024 * 1024
    field_separator = "\t"
    allele_delimiter = ","
    position_delimiter = "|"
    value_delimiter = ";"
    empty_field_char = "-"

    t = tarfile.open("/home/ec2-user/bystro/search2/13073_2016_396_moesm2_esm-2.tar")
    for member in t.getmembers():
        if 'annotation.tsv.gz' in member.name:
            ann_fh = t.extractfile(member)
            decompressed_data = gzip.GzipFile(fileobj=ann_fh, mode='rb')

    header_fields = decompressed_data.readline().rstrip(b"\n").decode('utf-8').split(field_separator)
    paths = [field.split(".") for field in header_fields]
    
    # boolean_map = {}
    # for field in osearch_map_config["booleanFields"]:
    #     boolean_map[field] = True
    boolean_map = {'discordant': True}
    row_idx = 0
    for line in decompressed_data:
        row = line.decode('utf-8').strip("\n").split(field_separator)

        row_document = {}
            
        for i, field in enumerate(row):
            allele_values = []
            for allele_value in field.split(allele_delimiter):
                if allele_value == empty_field_char:
                    allele_values.append(None)
                    continue
                
                position_values = []
                for pos_value in allele_value.split(position_delimiter):
                    if pos_value == empty_field_char:
                        position_values.append(None)
                        continue
                    
                    values = []
                    values_raw = pos_value.split(value_delimiter)
                    for value in values_raw:
                        if value == empty_field_char:
                            values.append(None)
                            continue
                        
                        if header_fields[i] in boolean_map:
                            if value == "1" or value == "True":
                                values.append(True)
                            elif value == "0" or value == "False":
                                values.append(False)
                            else:
                                raise ValueError(f"Encountered boolean value that wasn't encoded as 0/1 or True/False in field {field}, row {i}, value {value}")
                        else:
                            values.append(value)
                    
                    if len(values_raw) > 1:
                        position_values.append(values)
                    else:
                        position_values.append(values[0])
                
                allele_values.append(position_values)
            
            row_document = populate_hash_path_2(row_document, paths[i], allele_values)
        
        row_document_json = json.dumps(row_document)
        print(row_document_json)
            
            # # Assuming `indexer` is already defined and configured, replace this with the actual indexer instance
            # err = indexer.add(
            #     document_id=row_idx,
            #     body=row_document_json,
            #     on_failure=lambda ctx, item, res, err: print("ERROR:", err if err else f"{res['error']['type']}: {res['error']['reason']}")
            # )
            
            # if err:
            #     raise Exception(f"Unexpected error: {err}")
            
            # row_idx += 1

if __name__ == "__main__":
    main()