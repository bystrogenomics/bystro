from bystro.search.save.handler import _make_output_string

delims = {
    "empty_field": "!",
    "overlap": "/",
    "value": ";",
    "position": "|",
    "field": "\t",
}

def test_basic_functionality():
    rows = [
        # row 1
        [
            # column1
            [  # position values
                [  # value values
                    ["gene1_mrna1", None, "gene1_mrna2"], # overlap values
                    # gene1 has 3 transcripts, 1 of which is non-coding and doens't have an mrna record
                    ["gene1"]
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
                [
                    "row2_scalar"
                ]
            ]
        ]
    ]

    expected = (
        b"gene1_mrna1/!/gene1_mrna2;gene1|position2a;position2b\tcol2_scalar\nrow2_scalar\n"
    )

    assert _make_output_string(rows, delims) == expected

def test_empty_list():
    rows = []
    expected = b"\n"
    assert _make_output_string(rows, delims) == expected