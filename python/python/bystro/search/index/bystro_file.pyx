cimport cython
import tarfile
import gzip
from libc.stdint cimport uint32_t

cdef inline populate_hash_path(dict row_document, list field_path, list field_value):
    cdef dict current_dict = row_document
    cdef uint32_t i = 0
    for key in field_path:
        i += 1

        if key not in current_dict:
            current_dict[key] = {}
        if i == len(field_path):
            current_dict[key] = field_value
        else:
            current_dict = current_dict[key]

cdef class ReadAnnotationTarball:
    cdef:
        str index_name
        int chunk_size
        str field_separator
        str position_delimiter
        str overlap_delimiter
        str value_delimiter
        str empty_field_char
        object decompressed_data
        list header_fields
        list paths
        int id

    def __cinit__(self, str index_name,  dict delimiters, str tar_path, str annotation_name = 'annotation.tsv.gz', int chunk_size=500):
        self.index_name = index_name
        self.chunk_size = chunk_size
        self.field_separator = delimiters['field']
        self.position_delimiter = delimiters['position']
        self.overlap_delimiter = delimiters['overlap']
        self.value_delimiter = delimiters['value']
        self.empty_field_char = delimiters['empty_field']

        t = tarfile.open(tar_path)
        for member in t.getmembers():
            if annotation_name in member.name:
                ann_fh = t.extractfile(member)
                self.decompressed_data = gzip.GzipFile(fileobj=ann_fh, mode='rb')

        self.header_fields = self.decompressed_data.readline().rstrip(b"\n").decode('utf-8').split(self.field_separator)
        self.paths = [field.split(".") for field in self.header_fields]
        self.id = 0

    def __iter__(self):
        return self

    def __next__(self):
        cdef:
            bytes line
            list row
            list allele_values
            list position_values
            list values
            list overlap_values

        cdef list row_documents = []

        for line in self.decompressed_data:
            if not line:
                raise StopIteration

            _source = {}

            row = line.decode('utf-8').strip("\n").split(self.field_separator)
            for i, field in enumerate(row):
                if field == self.empty_field_char:
                    continue

                position_values = []
                for pos_value in field.split(self.position_delimiter):
                    if pos_value == self.empty_field_char:
                        position_values.append(None)
                        continue

                    values = []
                    values_raw = pos_value.split(self.value_delimiter)
                    for value in values_raw:
                        if value == self.empty_field_char:
                            values.append(None)
                            continue

                        overlap_values = []
                        for overlap_value in value.split(self.overlap_delimiter):
                            if overlap_value == self.empty_field_char:
                                overlap_values.append(None)
                                continue

                            overlap_values.append(overlap_value)

                        if len(overlap_values) == 1:
                            values.append(overlap_values[0])
                        else:
                            values.append(overlap_values)

                    position_values.append(values)

                populate_hash_path(_source, self.paths[i], position_values)

            if not _source:
                continue

            self.id += 1
            row_documents.append({"_index": self.index_name, "_id": self.id, "_source": _source})
            if len(row_documents) >= self.chunk_size:
                return row_documents

        if not row_documents:
            raise StopIteration

        return row_documents

    def get_header_fields(self):
        return self.header_fields

cpdef ReadAnnotationTarball read_annotation_tarball(str index_name,  dict delimiters, str tar_path, str annotation_name = 'annotation.tsv.gz', int chunk_size=500):
    return ReadAnnotationTarball(index_name, delimiters, tar_path, annotation_name, chunk_size)