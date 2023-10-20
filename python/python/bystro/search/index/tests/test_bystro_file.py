import pytest
import tarfile
import io
import gzip
import tempfile
from bystro.search.index.bystro_file import (  # type: ignore # pylint: disable=no-name-in-module,import-error  # noqa: E501
    read_annotation_tarball,
)
from bystro.search.utils.annotation import get_delimiters


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
    delims = get_delimiters()
    dv = delims["value"]  # e.g. ;
    df = delims["field"]  # e.g. \t
    do = delims["overlap"]  # e.g. chr(31)
    dp = delims["position"]  # e.g. |

    header = f"field1{df}field2{df}field3\n"
    field1val = f"value1a{dv}value1b{dp}value2aa{do}value2ab{dv}value2b{df}"
    field2val = f"value3a{dv}value3b{df}"
    field3val = f"value4a{dp}value4b\n"

    annotation_content = header + field1val + field2val + field3val

    mock_tarball_path = create_mock_tarball(annotation_content)

    reader = read_annotation_tarball(
        index_name="test_index",
        tar_path=mock_tarball_path,
        delimiters=delims,
        chunk_size=1,
    )

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

    result_data = next(reader)
    assert result_data == expected_data

    # Test the end of the data
    with pytest.raises(StopIteration):
        next(reader)

    # Test header fields
    assert reader.get_header_fields() == ["field1", "field2", "field3"]


if __name__ == "__main__":
    pytest.main()
