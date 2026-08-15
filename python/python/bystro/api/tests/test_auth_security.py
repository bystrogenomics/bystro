from __future__ import annotations

import stat

from msgspec import json

from bystro.api import auth


class FakeResponse:
    status_code = 200
    reason = "OK"
    text = json.encode({"bystro_access_token": "secret-dashboard-jwt"}).decode()

    def json(self) -> dict[str, str]:
        return {"bystro_access_token": "secret-dashboard-jwt"}


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append((url, kwargs))
        return FakeResponse()


def test_login_defaults_to_bystro_cloud_and_never_prints_token(
    monkeypatch,
    capsys,
) -> None:
    session = FakeSession()
    monkeypatch.setattr(auth.requests, "Session", lambda: session)
    monkeypatch.setattr(auth, "save_state", lambda *_args, **_kwargs: None)

    state = auth.login("scientist@example.com", "password", print_result=True)

    assert state.url == "https://bystro.cloud"
    assert state.access_token == "secret-dashboard-jwt"
    assert session.calls[0][0] == "https://bystro.cloud/api/user/auth/local"
    output = capsys.readouterr().out
    assert "Login successful" in output
    assert "secret-dashboard-jwt" not in output


def test_save_state_is_atomic_private_and_redacted(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    credential_dir = tmp_path / ".bystro"
    credential_path = credential_dir / "bystro_authentication_token.json"
    monkeypatch.setattr(auth, "DEFAULT_DIR", str(credential_dir))
    monkeypatch.setattr(auth, "CREDENTIALS_PATH", str(credential_path))
    state = auth.CachedAuth(
        email="scientist@example.com",
        access_token="top-secret",
        url="https://bystro.cloud",
    )

    auth.save_state(state, print_result=True)

    assert auth.load_state() == state
    assert stat.S_IMODE(credential_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(credential_path.stat().st_mode) == 0o600
    output = capsys.readouterr().out
    assert "[redacted]" in output
    assert "top-secret" not in output


def test_normalize_url_avoids_default_port_noise() -> None:
    assert auth.normalize_url("https://bystro.cloud/") == "https://bystro.cloud"
    assert auth.normalize_url("localhost", 8080) == "http://localhost:8080"
    assert auth.normalize_url("http://127.0.0.1:9005") == "http://127.0.0.1:9005"
    assert auth.normalize_url("[::1]:8080") == "http://[::1]:8080"
