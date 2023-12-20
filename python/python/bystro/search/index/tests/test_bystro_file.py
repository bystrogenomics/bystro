import pytest
import tarfile
import io
import gzip
import tempfile
from bystro.search.index.bystro_file import (  # type: ignore # pylint: disable=no-name-in-module,import-error  # noqa: E501
    read_annotation_tarball,
)
from bystro.search.utils.annotation import DelimitersConfig


def create_mock_tarball(annotation_content):
    # Create a compressed annotation file
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as f:
        f.write(annotation_content.encode())

    # Create a temporary tar file
    temp_tar_file = tempfile.NamedTemporaryFile(delete=False)
    with tarfile.open(fileobj=temp_tar_file, mode="w") as t:
        tarinfo = tarfile.TarInfo(name="annotation.tsv.gz")
        tarinfo.size = len(buffer.getvalue())
        buffer.seek(0)
        t.addfile(tarinfo, buffer)

    return temp_tar_file.name


def test_read_annotation_tarball():
    delims = DelimitersConfig()
    delim_v = delims.value  # e.g. ;
    delim_f = delims.field  # e.g. \t
    delim_o = delims.overlap  # e.g. /
    delim_p = delims.position  # e.g. |

    header = f"field1{delim_f}field2{delim_f}field3\n"
    field1val = f"value1a{delim_v}value1b{delim_p}value2aa{delim_o}value2ab{delim_v}value2b{delim_f}"
    field2val = f"value3a{delim_v}value3b{delim_f}"
    field3val = f"value4a{delim_p}value4b\n"

    annotation_content = header + field1val + field2val + field3val

    mock_tarball_path = create_mock_tarball(annotation_content)

    reader = read_annotation_tarball(
        index_name="test_index",
        tar_path=mock_tarball_path,
        delimiters=delims,
        chunk_size=1,
    )

    # In the expected data commented explanations that follow
    # the overlap delimiter is considered to be "/"
    expected_data = [
        {
            "_index": "test_index",
            "_id": 1,
            "_source": {
                "field1": [
                    # value1a;value1b|value2aa/value2ab;value2b
                    [["value1a"], ["value1b"]], # value1a;value1b
                    [["value2aa","value2ab"], ["value2b"]], # value2aa/value2ab;value2b
                ],
                "field2": [
                    [["value3a"], ["value3b"]]  #value3a;value3B
                ],
                "field3": [
                    [["value4a"]], [["value4b"]],  #value4a|value4B
                ],
            },
        }
    ]
    import numpy as np
    import json
    result_data = next(reader)
    assert result_data == expected_data
    np.testing.assert_equal(result_data, expected_data)
    print(json.dumps(expected_data, indent=4))

    # Test the end of the data
    with pytest.raises(StopIteration):
        next(reader)

    # Test header fields
    assert reader.get_header_fields() == ["field1", "field2", "field3"]


if __name__ == "__main__":
    pytest.main()
