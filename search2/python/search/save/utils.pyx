def make_output_string(rows: list, delims: dict):
    empty_field_char = delims['miss']
    for row_idx, row in enumerate(rows):
        # Some fields may just be missing; we won't store even the alt/pos [[]] structure for those
        for i, column in enumerate(row):
            if column is None:
                row[i] = empty_field_char
                continue

            # For now, we don't store multiallelics; top level array is placeholder only
            # With breadth 1
            if not isinstance(column, list):
                row[i] = str(column)
                continue

            for j, position_data in enumerate(column):
                if position_data is None:
                    column[j] = empty_field_char
                    continue

                if isinstance(position_data, list):
                    inner_values = []
                    for sub in position_data:
                        if sub is None:
                            inner_values.append(empty_field_char)
                            continue

                        if isinstance(sub, list):
                            inner_values.append(delims['value'].join(map(lambda x: str(x) if x is not None else empty_field_char, sub)))
                        else:
                            inner_values.append(str(sub))

                    column[j] = delims['pos'].join(inner_values)

            row[i] = delims['overlap'].join(column)

        rows[row_idx] = delims['fieldSep'].join(row)

    return "\n".join(rows) + "\n"

def cython_hello():
    print("HELLO WORLD")