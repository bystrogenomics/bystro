cimport cython
import tarfile
import gzip
from typing import List
from libc.stdint cimport uint32_t

cdef inline dict populate_hash_path(dict row_document, list field_path, field_value):
    cdef dict current = row_document
    cdef uint32_t path_len = len(field_path)
    cdef uint32_t k

    for k in range(path_len):
        key = field_path[k]

        if key not in row_document:
            current[key] = {}

        if k < path_len - 1:
            current = current[key]
        else:
            current[key] = field_value

    return row_document

cdef class ReadAnnotationTarball:
    cdef:
        str index_name
        int chunk_size
        str field_separator
        str allele_delimiter
        str position_delimiter
        str value_delimiter
        str empty_field_char
        object decompressed_data
        list header_fields
        list paths
        dict boolean_map
        dict idx_action
        dict row_document

    def __cinit__(self, str index_name,  dict boolean_map, dict delimiters, str tar_path, str annotation_name = 'annotation.tsv.gz', int chunk_size=500):
        self.index_name = index_name
        self.chunk_size = chunk_size
        self.field_separator = delimiters['field']
        self.allele_delimiter = delimiters['allele']
        self.position_delimiter = delimiters['position']
        self.value_delimiter = delimiters['value']
        self.empty_field_char = delimiters['empty_field']

        t = tarfile.open(tar_path)
        for member in t.getmembers():
            if annotation_name in member.name:
                ann_fh = t.extractfile(member)
                self.decompressed_data = gzip.GzipFile(fileobj=ann_fh, mode='rb')

        self.header_fields = self.decompressed_data.readline().rstrip(b"\n").decode('utf-8').split(self.field_separator)
        self.paths = [field.split(".") for field in self.header_fields]
        self.boolean_map = boolean_map
        self.idx_action = {"_index": self.index_name, "_id": 0, "_source": {}}

    def __iter__(self):
        return self

    def __next__(self):
        cdef bytes line
        cdef list row
        cdef list allele_values
        cdef list position_values
        cdef list values

        line = self.decompressed_data.readline()
        if not line:
            raise StopIteration

        row = line.decode('utf-8').strip("\n").split(self.field_separator)

        self.idx_action['_id'] += 1
        self.idx_action['_source'] = {}
        for i, field in enumerate(row):
            allele_values = []
            for allele_value in field.split(self.allele_delimiter):
                if allele_value == self.empty_field_char:
                    allele_values.append(None)
                    continue

                position_values = []
                for pos_value in allele_value.split(self.position_delimiter):
                    if pos_value == self.empty_field_char:
                        position_values.append(None)
                        continue

                    values = []
                    values_raw = pos_value.split(self.value_delimiter)
                    for value in values_raw:
                        if value == self.empty_field_char:
                            values.append(None)
                            continue

                        if self.header_fields[i] in self.boolean_map:
                            if value == "1" or value == "True":
                                values.append(True)
                            elif value == "0" or value == "False":
                                values.append(False)
                            else:
                                raise ValueError(
                                    f"Encountered boolean value that wasn't encoded as 0/1 or True/False in field {field}, row {i}, value {value}")
                        else:
                            values.append(value)

                    if len(values_raw) > 1:
                        position_values.append(values)
                    else:
                        position_values.append(values[0])

                allele_values.append(position_values)

            self.idx_action['_source'] = populate_hash_path(self.idx_action['_source'], self.paths[i], allele_values)

        return self.idx_action

cpdef ReadAnnotationTarball read_annotation_tarball(str index_name,  dict boolean_map, dict delimiters, str tar_path, str annotation_name = 'annotation.tsv.gz', int chunk_size=500):
    return ReadAnnotationTarball(index_name, boolean_map, delimiters, tar_path, annotation_name, chunk_size)