"""Utility functions for Opensearch 2.*"""


def gather_opensearch_args(search_conf: dict):
    """
    Return arguments required to instantiate OpenSearch Client
    Supports both https and http protocols
    When hosts are prefixed https://, http_auth, client_cert, client_key, ca_certs,
         verify_certs, ssl_assert_hostname, and ssl_show_warn will be in effect
    """
    return dict(
        hosts=list(search_conf["connection"]["nodes"]),
        http_compress=True,
        timeout=search_conf["connection"].get("request_timeout", 600),
        http_auth=search_conf["auth"].get("auth"),
        client_cert=search_conf["auth"].get("client_cert_path"),
        client_key=search_conf["auth"].get("client_key_path"),
        ca_certs=search_conf["auth"].get("ca_certs_path"),
        verify_certs=True,
        ssl_assert_hostname=True,
        ssl_show_warn=True,
        pool_maxsize = 16
    )
