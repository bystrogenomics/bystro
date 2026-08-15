from __future__ import annotations

from datetime import datetime, timezone

from msgspec import json as mjson
import pytest

from bystro.api import auth


class FakeResponse:
    def __init__(self, status_code: int, payload: object, reason: str = "OK") -> None:
        self.status_code = status_code
        self._payload = payload
        self.reason = reason

    def json(self) -> object:
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, object]]] = []
        self.closed = False

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append(("POST", url, kwargs))
        return self.responses.pop(0)

    def put(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append(("PUT", url, kwargs))
        return self.responses.pop(0)


def test_token_response_models_decode_dashboard_field_without_breaking_public_attribute() -> None:
    encoded = b'{"bystro_access_token":"dashboard-token"}'

    login_response = mjson.decode(encoded, type=auth.LoginResponse)
    signup_response = mjson.decode(encoded, type=auth.SignupResponse)

    assert login_response.access_token == "dashboard-token"
    assert signup_response.access_token == "dashboard-token"


def test_signup_uses_site_gate_session_and_current_signed_consent(monkeypatch) -> None:
    session = FakeSession(
        [
            FakeResponse(200, {"success": True}),
            FakeResponse(200, {"bystro_access_token": "signup-token"}),
        ]
    )
    monkeypatch.setattr(auth.requests, "Session", lambda: session)
    consent = auth.LegalConsent(
        over_18=True,
        accepted_terms=True,
        accepted_privacy=True,
        accepted_user_acknowledgment=True,
        signature_name="Test User",
    )

    state = auth.signup(
        "test@example.com",
        "password",
        "Test User",
        host="https://example.test",
        cache=False,
        legal_consent=consent,
        site_access_code="invite-code",
    )

    assert state.access_token == "signup-token"
    assert session.closed is True
    assert session.calls[0] == (
        "POST",
        "https://example.test/api/site-gate/authenticate",
        {
            "json": {"password": "invite-code"},
            "timeout": 30.0,
            "allow_redirects": False,
        },
    )
    signup_call = session.calls[1]
    assert signup_call[0:2] == ("PUT", "https://example.test/api/user")
    assert signup_call[2]["allow_redirects"] is False
    assert signup_call[2]["data"] == {
        "email": "test@example.com",
        "name": "Test User",
        "password": "password",
        "over18": "true",
        "acceptedTerms": "true",
        "acceptedPrivacy": "true",
        "acceptedUserAcknowledgment": "true",
        "consentName": "Test User",
    }


def test_login_presents_site_access_code_without_caching_it(monkeypatch) -> None:
    session = FakeSession(
        [
            FakeResponse(200, {"success": True}),
            FakeResponse(200, {"bystro_access_token": "login-token"}),
        ]
    )
    monkeypatch.setattr(auth.requests, "Session", lambda: session)

    state = auth.login(
        "test@example.com",
        "password",
        host="https://example.test",
        cache=False,
        site_access_code="invite-code",
    )

    assert state == auth.CachedAuth(
        email="test@example.com",
        url="https://example.test",
        access_token="login-token",
    )
    assert "invite-code" not in repr(state)
    assert [call[1] for call in session.calls] == [
        "https://example.test/api/site-gate/authenticate",
        "https://example.test/api/user/auth/local",
    ]
    assert all(call[2]["allow_redirects"] is False for call in session.calls)


def test_login_surfaces_structured_legal_consent_requirement(monkeypatch) -> None:
    session = FakeSession(
        [
            FakeResponse(
                428,
                {
                    "error": "legal_consent_required",
                    "missing": ["over18", "acceptedTerms", "acceptedPrivacy"],
                    "message": "server copy must not become the stable SDK contract",
                },
                "Precondition Required",
            )
        ]
    )
    monkeypatch.setattr(auth.requests, "Session", lambda: session)

    with pytest.raises(auth.LegalConsentRequiredError) as raised:
        auth.login(
            "test@example.com",
            "password",
            host="https://example.test",
            cache=False,
        )

    assert raised.value.status_code == 428
    assert raised.value.code == "legal_consent_required"
    assert raised.value.missing == (
        "over18",
        "acceptedTerms",
        "acceptedPrivacy",
    )
    assert "server copy" not in str(raised.value)


@pytest.fixture
def isolated_credentials(tmp_path, monkeypatch):
    credential_dir = tmp_path / ".bystro"
    credential_path = credential_dir / auth.STATE_FILE
    monkeypatch.setattr(auth, "DEFAULT_DIR", str(credential_dir))
    monkeypatch.setattr(auth, "CREDENTIALS_PATH", str(credential_path))
    return credential_path


def test_load_state_existing_file(isolated_credentials) -> None:
    assert isolated_credentials.name == auth.STATE_FILE
    state = auth.CachedAuth(
        email="scientist@example.com",
        url="https://bystro.cloud",
        access_token="token",
    )
    auth.save_state(state)

    assert auth.load_state() == state


def test_load_state_hardens_legacy_cache_permissions(isolated_credentials) -> None:
    isolated_credentials.parent.mkdir(mode=0o755, parents=True)
    state = auth.CachedAuth(
        email="scientist@example.com",
        url="https://bystro.cloud",
        access_token="legacy-token",
    )
    isolated_credentials.write_bytes(mjson.encode(state))
    isolated_credentials.chmod(0o644)

    assert auth.load_state() == state
    assert isolated_credentials.parent.stat().st_mode & 0o777 == 0o700
    assert isolated_credentials.stat().st_mode & 0o777 == 0o600


def test_cached_auth_representations_redact_bearer_token() -> None:
    state = auth.CachedAuth(
        email="scientist@example.com",
        url="https://bystro.cloud",
        access_token="secret-dashboard-jwt",
    )

    assert "secret-dashboard-jwt" not in repr(state)
    assert "[redacted]" in repr(state)
    assert state.__rich_repr__() == (
        ("email", "scientist@example.com"),
        ("access_token", "[redacted]"),
        ("url", "https://bystro.cloud"),
    )


def test_load_state_no_file(isolated_credentials) -> None:
    assert not isolated_credentials.exists()
    assert auth.load_state() is None


def test_load_state_rejects_corrupt_cache_with_typed_error(isolated_credentials) -> None:
    isolated_credentials.parent.mkdir(parents=True)
    isolated_credentials.write_text("not-json")

    with pytest.raises(auth.AuthenticationError, match="state is invalid"):
        auth.load_state()


@pytest.mark.parametrize("status_code", [404, 500])
def test_login_failure(monkeypatch, status_code: int) -> None:
    session = FakeSession([FakeResponse(status_code, {"detail": "rejected"}, "Error")])
    monkeypatch.setattr(auth.requests, "Session", lambda: session)

    with pytest.raises(auth.AuthenticationError) as raised:
        auth.login(
            "test@example.com",
            "password",
            "http://localhost",
            8080,
            cache=False,
        )

    assert raised.value.status_code == status_code
    assert "rejected" not in str(raised.value)


def test_signup_uses_current_dashboard_token_field(monkeypatch) -> None:
    session = FakeSession([FakeResponse(200, {"bystro_access_token": "signup-token"})])
    monkeypatch.setattr(auth.requests, "Session", lambda: session)

    state = auth.signup(
        "test@example.com",
        "password",
        "Test User",
        "http://localhost",
        8080,
        cache=False,
    )

    assert state == auth.CachedAuth(
        email="test@example.com",
        url="http://localhost:8080",
        access_token="signup-token",
    )
    assert session.calls[0][0:2] == (
        "PUT",
        "http://localhost:8080/api/user",
    )


def test_get_user(monkeypatch) -> None:
    state = auth.CachedAuth(
        email="scientist@example.com",
        url="https://bystro.cloud",
        access_token="token",
    )
    payload = {
        "_id": "user-1",
        "options": {"autoUploadToS3": False},
        "name": "Scientist",
        "email": "scientist@example.com",
        "accounts": ["bystro"],
        "role": "user",
        "lastLogin": "2023-09-06T05:45:01.446Z",
    }
    calls: list[tuple[str, object]] = []

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        calls.append((url, kwargs))
        return FakeResponse(200, payload)

    monkeypatch.setattr(auth.requests, "get", fake_get)

    profile = auth.get_user(cached_auth=state)

    assert profile == auth.UserProfile(
        options={"autoUploadToS3": False},
        _id="user-1",
        name="Scientist",
        email="scientist@example.com",
        accounts=["bystro"],
        role="user",
        last_login=datetime(2023, 9, 6, 5, 45, 1, 446000, tzinfo=timezone.utc),
    )
    request_options = calls[0][1]
    assert isinstance(request_options, dict)
    assert request_options["headers"] == {"Authorization": "Bearer token"}
    assert request_options["allow_redirects"] is False


def test_authenticate_accepts_explicit_state() -> None:
    state = auth.CachedAuth(
        email="scientist@example.com",
        url="https://bystro.cloud",
        access_token="token",
    )

    assert auth.authenticate(state) == (state, {"Authorization": "Bearer token"})


def test_authenticate_without_login_raises(isolated_credentials) -> None:
    assert not isolated_credentials.exists()
    with pytest.raises(auth.AuthenticationError, match="not logged in"):
        auth.authenticate()
