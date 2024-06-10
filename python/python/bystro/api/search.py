from typing import Optional, Mapping, Any, Union, Collection
from requests.auth import AuthBase

from opensearchpy import OpenSearch, RequestsHttpConnection, AsyncOpenSearch, AsyncHttpConnection

from bystro.api.auth import CachedAuth

QUERY_ENDPOINT = "/api/jobs/{job_id}/opensearch"


class JWTAuth(AuthBase):
    """
    A class that provides the JWT authentication for the OpenSearch connection.

    Parameters
    ----------
    token : str
        The JWT token.

    Attributes
    ----------
    token : str
        The JWT token.

    Methods
    -------
    __call__(r, *args, **kwargs)
        Add the Authorization header to the request.
    """

    def __init__(self, token):
        self.token = token

    def __call__(self, r, *_args, **_kwargs):
        r.headers["Authorization"] = f"Bearer {self.token}"
        return r


class BystroProxyHttpConnection(RequestsHttpConnection):
    """
    A class that provides the HTTP connection to the OpenSearch server.

    Parameters
    ----------
    path_prefix : str
        The path prefix to use when connecting to the OpenSearch server.

    Attributes
    ----------
    path_prefix : str
        The path prefix to use when connecting to the OpenSearch server.

    Methods
    -------
    perform_request(method, path, params=None, body=None,
                    headers=None, timeout=None, ignore=(), **kwargs)
        Perform a request to the OpenSearch server.
    """

    def __init__(self, *args, path_prefix="", **kwargs):
        self.path_prefix = path_prefix
        super().__init__(*args, **kwargs)
        self.auth = JWTAuth(kwargs.get("http_auth"))

    def perform_request(
        self, method, url, params=None, body=None, headers=None, timeout=None, ignore=(), **kwargs
    ):
        """
        Perform a request to the OpenSearch server.

        Parameters
        ----------
        method : str
            The HTTP method to use.
        url : str
            The URL to connect to.
        params : Optional[Mapping[str, Any]]
            The parameters to include in the request.
        body : Optional[bytes]
            The body of the request.
        headers : Optional[Mapping[str, str]]
            The headers to include in the request.
        timeout : Optional[Union[int, float]]
            The timeout for the request.
        ignore : Collection[int]
            The status codes to ignore.
        """
        # Ensure path_prefix is included in the URL
        if not url.startswith(self.path_prefix):
            url = self.path_prefix + url

        return super().perform_request(
            method,
            url,
            params=params,
            body=body,
            headers=headers,
            timeout=timeout,
            ignore=ignore,
            **kwargs,
        )


class AsyncConnectorJWTAuth:
    """
    A class that provides the JWT authentication for the OpenSearch connection.

    Parameters
    ----------
    token : str
        The JWT token.
    """

    def __init__(self, token):
        self.token = token

    def __call__(self, _method, _url, _query_string: str, _body=None):
        return {"Authorization": f"Bearer {self.token}"}


class BystroProxyAsyncHttpConnection(AsyncHttpConnection):
    """
    An OpenSearch connection class that uses a proxy to connect to the OpenSearch server.

    Parameters
    ----------
    path_prefix : str
        The path prefix to use when connecting to the OpenSearch server.

    Attributes
    ----------
    path_prefix : str
        The path prefix to use when connecting to the OpenSearch server.

    Methods
    -------
    perform_request(method, url, params=None, body=None, timeout=None, ignore=(), headers=None)
        Perform a request to the OpenSearch server.
    """

    def __init__(self, *args, path_prefix: str = "", **kwargs):
        self.path_prefix = path_prefix
        super().__init__(*args, **kwargs)
        self.auth = kwargs.get("http_auth")

    async def perform_request(
        self,
        method: str,
        url: str,
        params: Optional[Mapping[str, Any]] = None,
        body: Optional[bytes] = None,
        timeout: Optional[Union[int, float]] = None,
        ignore: Collection[int] = (),
        headers: Optional[Mapping[str, str]] = None,
    ):
        """
        Perform a request to the OpenSearch server.

        Parameters
        ----------
        method : str
            The HTTP method to use.
        url : str
            The URL to connect to.
        params : Optional[Mapping[str, Any]]
            The parameters to include in the request.
        body : Optional[bytes]
            The body of the request.
        timeout : Optional[Union[int, float]]
            The timeout for the request.
        ignore : Collection[int]
            The status codes to ignore.
        headers : Optional[Mapping[str, str]]
            The headers to include in the request.
        """
        if not url.startswith(self.path_prefix):
            path = self.path_prefix + url

        return await super().perform_request(
            method=method,
            url=path,
            params=params,
            body=body,
            headers=headers,
            timeout=timeout,
            ignore=ignore,
        )


def get_async_proxied_opensearch_client(auth: CachedAuth, job_id: str, client_args: dict | None = None):
    """
    Create an OpenSearch client that uses a proxy to connect to the OpenSearch server.

    Parameters
    ----------
    auth : CachedAuth
        The authentication information.
    job_id : str
        The ID of the job to connect to.

    Returns
    -------
    AsyncOpenSearch
        The AsyncOpenSearch client.
    """
    [protocol, host] = auth.url.split("://")

    [host, port] = host.split(":")

    if client_args is None:
        client_args = {}

    # Initialize OpenSearch client with custom connection class

    client = AsyncOpenSearch(
        hosts=[{"host": host, "port": int(port)}],
        use_ssl=bool(protocol == "https" or int(port) == 443),
        connection_class=BystroProxyAsyncHttpConnection,
        path_prefix=QUERY_ENDPOINT.format(job_id=job_id),
        http_auth=AsyncConnectorJWTAuth(auth.access_token),
        http_compress=True,
        timeout=client_args.get("timeout", 1200),
        pool_maxsize=client_args.get("pool_maxsize", 16),
    )

    return client


def get_proxied_opensearch_client(auth: CachedAuth, job_id: str, client_args: dict | None = None):
    """
    Create an OpenSearch client that uses a proxy to connect to the OpenSearch server.

    Parameters
    ----------
    auth : CachedAuth
        The authentication information.
    job_id : str
        The ID of the job to connect to.

    Returns
    -------
    OpenSearch
        The OpenSearch client.
    """
    [protocol, host] = auth.url.split("://")

    [host, port] = host.split(":")

    if client_args is None:
        client_args = {}

    # Initialize OpenSearch client with custom connection class
    client = OpenSearch(
        hosts=[{"host": host, "port": int(port)}],
        use_ssl=bool(protocol == "https" or int(port) == 443),
        connection_class=BystroProxyHttpConnection,
        path_prefix=QUERY_ENDPOINT.format(job_id=job_id),
        http_auth=JWTAuth(auth.access_token),
        http_compress=True,
        timeout=client_args.get("timeout", 1200),
        pool_maxsize=client_args.get("pool_maxsize", 16),
    )

    return client
