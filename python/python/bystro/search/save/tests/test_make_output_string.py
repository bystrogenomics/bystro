from bystro.search.save.handler import _make_output_string, _populate_data
from bystro.search.utils.annotation import DelimitersConfig

delims = DelimitersConfig(
    empty_field="!",
    overlap="/",
    value=";",
    position="|",
    field="\t",
)

def test_populate_data():
    data = {"a": {"b": {"c": "value"}}}

    # Positive cases
    assert _populate_data(["a", "b", "c"], data) == "value"
    assert _populate_data(["a"], data) == {"b": {"c": "value"}}

    # Negative cases
    assert _populate_data(["a", "b", "d"], data) is None
    assert _populate_data("a.b.c", data) is None
    assert _populate_data(["a", "b", "c"], None) is None
    assert _populate_data(["a.exact"], data) is None
    assert _populate_data(["a", "b.exact"], data) is None


def test_populate_data_real_example_nested():
    _real_example_nested = {
        "nearest": {
            "refSeq": {
                "name2": [[["GCFC2"]]],
                "name": [[["NM_001201334"], ["NM_003203"]]],
                "dist": [[["0"]]],
            }
        }
    }

    assert _populate_data(["nearest", "refSeq", "name2"], _real_example_nested) == [
        [["GCFC2"]]
    ]
    assert _populate_data(["nearest", "refSeq", "name"], _real_example_nested) == [
        [["NM_001201334"], ["NM_003203"]]
    ]
    assert _populate_data(["nearest", "refSeq", "dist"], _real_example_nested) == [
        [["0"]]
    ]


def test_populate_data_real_example_scalar():
    _real_example_scalar = {
        "chrom": [[["chr2"]]],
        "pos": [[["75928300"]]],
        "type": [[["SNP"]]]
    }

    assert _populate_data(["chrom"], _real_example_scalar) == [[["chr2"]]]
    assert _populate_data(["pos"], _real_example_scalar) == [[["75928300"]]]
    assert _populate_data(["type"], _real_example_scalar) == [[["SNP"]]]

    # _populate_data considers any non-dict value a leaf value and will return it as is
    # and the provided "path" is a label of the leaf's data, a no-op
    assert _populate_data("chrom", [[["chr2"]]]) == [[["chr2"]]]
    assert _populate_data("pos", [[["75928300"]]]) == [[["75928300"]]]
    assert _populate_data("type", [[["SNP"]]]) == [[["SNP"]]]


def test_basic_functionality():
    rows = [
        # row 1
        [
            # column1
            [  # position values
                [  # value values
                    ["gene1_mrna1", None, "gene1_mrna2"],  # overlap values
                    # gene1 has 3 transcripts, 1 of which is non-coding and doens't have an mrna record
                    ["gene1"],
                ],
                # 2 value delimited values at the next position in the indel
                ["position2a", "position2b"],
            ],
            # column 2
            [  # position values
                [
                    ["col2_scalar"],
                ]
            ],
        ],
        # row2
        [
            # column1
            [
                # Retain backwards compat with scalar values in 2nd dimension
                ["row2_scalar"]
            ]
        ],
    ]

    expected = b"gene1_mrna1/!/gene1_mrna2;gene1|position2a;position2b\tcol2_scalar\nrow2_scalar\n"

    assert _make_output_string(rows, delims) == expected


def test_empty_list():
    rows: list = []
    expected = b"\n"
    assert _make_output_string(rows, delims) == expected
