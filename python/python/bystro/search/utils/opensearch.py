"""Utility functions for Opensearch 2.*"""


def gather_opensearch_args(search_conf: dict):
    """
    Return arguments required to instantiate OpenSearch Client
    Supports both https and http protocols
    When hosts are prefixed https://, http_auth, client_cert, client_key, ca_certs,
         verify_certs, ssl_assert_hostname, and ssl_show_warn will be in effect
    """
    http_auth = None
    client_cert = None
    client_key = None
    ca_certs = None

    if "auth" in search_conf:
        username = search_conf["auth"]["username"]
        password = search_conf["auth"]["password"]
        http_auth = (username, password)
        client_cert = search_conf["auth"].get("client_cert_path")
        client_key = search_conf["auth"].get("client_key_path")
        ca_certs = search_conf["auth"].get("ca_certs_path")

    return dict(
        hosts=list(search_conf["connection"]["nodes"]),
        http_compress=True,
        timeout=search_conf["connection"].get("request_timeout", 1200),
        http_auth=http_auth,
        client_cert=client_cert,
        client_key=client_key,
        ca_certs=ca_certs,
        use_ssl=search_conf["connection"].get("use_ssl", False),
        verify_certs=search_conf["connection"].get("verify_certs", False),
        ssl_assert_hostname=True,
        ssl_show_warn=True,
        pool_maxsize=16,
    )
