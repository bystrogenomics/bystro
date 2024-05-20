import pytest
from unittest.mock import AsyncMock, Mock, patch

from opensearchpy import AsyncHttpConnection


from bystro.api.auth import CachedAuth
from bystro.api.search import (
    AsyncConnectorJWTAuth,
    BystroProxyAsyncHttpConnection,
    get_async_proxied_opensearch_client,
)


@pytest.fixture
def auth_token():
    return "test_token"


@pytest.fixture
def job_id():
    return "1234"


@pytest.fixture
def cached_auth(auth_token):
    return CachedAuth(email="email", access_token=auth_token, url="https://testhost:9200")


@pytest.mark.asyncio
async def test_async_connector_jwt_auth(auth_token):
    auth = AsyncConnectorJWTAuth(auth_token)
    headers = auth("GET", "http://testhost:9200", "")

    assert headers["Authorization"] == f"Bearer {auth_token}"


@pytest.mark.asyncio
async def test_bystro_proxy_async_http_connection(auth_token):
    connection = BystroProxyAsyncHttpConnection(
        http_auth=AsyncConnectorJWTAuth(auth_token), path_prefix="/testprefix"
    )

    assert connection.path_prefix == "/testprefix"
    assert isinstance(connection.auth, AsyncConnectorJWTAuth)


def test_get_async_proxied_opensearch_client(cached_auth, job_id):
    with (
        patch("bystro.api.search.AsyncOpenSearch", new_callable=Mock) as mock_client,
        patch("bystro.api.search.AsyncConnectorJWTAuth", new_callable=Mock) as mock_auth,
    ):
        client = get_async_proxied_opensearch_client(cached_auth, job_id)

        assert isinstance(client, Mock)
        mock_client.assert_called_once_with(
            hosts=[{"host": "testhost", "port": "9200"}],
            use_ssl=True,
            connection_class=BystroProxyAsyncHttpConnection,
            path_prefix=f"/api/jobs/{job_id}/opensearch",
            http_auth=mock_auth.return_value,  # Ensure the mock instance is used
        )



@pytest.mark.asyncio
async def test_bystro_proxy_async_http_connection_perform_request(auth_token):
    with patch.object(
        AsyncHttpConnection, "perform_request", new_callable=AsyncMock
    ) as mock_super_perform_request:
        connection = BystroProxyAsyncHttpConnection(
            http_auth=AsyncConnectorJWTAuth(auth_token), path_prefix="/testprefix"
        )

        await connection.perform_request("GET", "/testurl")

        mock_super_perform_request.assert_awaited_once_with(
            method="GET",
            url="/testprefix/testurl",
            params=None,
            body=None,
            headers=None,
            timeout=None,
            ignore=(),
        )
