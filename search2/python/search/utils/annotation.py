from typing import Optional

_default_delimiters = {
    "field": "\t",
    "allele": "/",
    "position": "|",
    'overlap': "\\",
    "value": ";",
    "empty_field": "!",
}

def get_delimiters(annotation_conf: Optional[dict] = None):
    if annotation_conf:
        return annotation_conf.get("delimiters", _default_delimiters)
    return _default_delimiters