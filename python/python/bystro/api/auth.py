"""Authentication helpers for the Bystro cloud API.

The dashboard issues a browser JWT named ``bystro_access_token``.  This module
stores that credential locally so the rest of the Python package, including
``bystro.think``, can use the same account as the dashboard.
"""

from __future__ import annotations

import datetime
from contextlib import suppress
import logging
import os
from pathlib import Path
import tempfile
from typing import Literal, cast
from urllib.parse import SplitResult, urlsplit, urlunsplit

from msgspec import DecodeError, Struct, ValidationError, json as mjson
import requests

logger = logging.getLogger(__name__)

DEFAULT_BYSTRO_URL = "https://bystro.cloud"
DEFAULT_DIR = os.path.expanduser("~/.bystro")
STATE_FILE = "bystro_authentication_token.json"
CREDENTIALS_PATH = os.path.join(DEFAULT_DIR, STATE_FILE)
_REQUEST_TIMEOUT_SECONDS = 30.0


class AuthenticationError(RuntimeError):
    """A dashboard authentication request was rejected or unavailable."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        missing: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.missing = missing


class LegalConsentRequiredError(AuthenticationError):
    """The dashboard requires a current legal-consent assertion."""


class SiteGateAuthenticationError(AuthenticationError):
    """The optional Bystro site-access code was rejected."""


class SignupResponse(Struct, rename={"access_token": "bystro_access_token"}):
    """Successful dashboard signup response."""

    access_token: str


class LoginResponse(Struct, rename={"access_token": "bystro_access_token"}):
    """Successful dashboard login response."""

    access_token: str


class CachedAuth(Struct):
    """Locally cached dashboard authentication state."""

    email: str
    access_token: str
    url: str

    def __repr__(self) -> str:
        return (
            f"CachedAuth(email={self.email!r}, access_token='[redacted]', "
            f"url={self.url!r})"
        )

    def __rich_repr__(self) -> tuple[tuple[str, object], ...]:
        return (
            ("email", self.email),
            ("access_token", "[redacted]"),
            ("url", self.url),
        )


class LegalConsent(Struct, frozen=True):
    """An explicit signed legal-consent assertion for signup or legacy login."""

    over_18: bool
    accepted_terms: bool
    accepted_privacy: bool
    accepted_user_acknowledgment: bool
    signature_name: str
    viewed_terms: bool = False
    viewed_privacy: bool = False

    def __post_init__(self) -> None:
        signature = self.signature_name.strip()
        if not signature:
            raise ValueError("signature_name cannot be empty")
        if len(signature) > 500:
            raise ValueError("signature_name cannot exceed 500 characters")

    @classmethod
    def accepted(
        cls,
        signature_name: str,
        *,
        viewed_terms: bool = False,
        viewed_privacy: bool = False,
    ) -> "LegalConsent":
        """Create an explicit assertion accepting all current legal terms."""

        return cls(
            over_18=True,
            accepted_terms=True,
            accepted_privacy=True,
            accepted_user_acknowledgment=True,
            signature_name=signature_name,
            viewed_terms=viewed_terms,
            viewed_privacy=viewed_privacy,
        )

    def form_fields(self) -> dict[str, str]:
        """Return the dashboard's authoritative legal-consent wire fields."""

        fields = {
            "over18": str(self.over_18).lower(),
            "acceptedTerms": str(self.accepted_terms).lower(),
            "acceptedPrivacy": str(self.accepted_privacy).lower(),
            "acceptedUserAcknowledgment": str(self.accepted_user_acknowledgment).lower(),
            "consentName": self.signature_name.strip(),
        }
        if self.viewed_terms:
            fields["viewedTerms"] = "true"
        if self.viewed_privacy:
            fields["viewedPrivacy"] = "true"
        return fields


class UserProfile(Struct, rename="camel"):
    """Bystro dashboard user profile."""

    _id: str
    options: dict[str, object]
    name: str
    email: str
    accounts: list[str]
    role: str
    last_login: datetime.datetime | None = None


def _url_with_inferred_scheme(host: str) -> str:
    try:
        hostname = urlsplit(f"//{host}").hostname
    except ValueError as exc:
        raise ValueError("host must include a valid hostname") from exc
    if hostname is None:
        raise ValueError("host must include a hostname")
    scheme = "http" if hostname in {"localhost", "127.0.0.1", "::1"} else "https"
    return f"{scheme}://{host}"


def _replace_port(parsed: SplitResult, port: int) -> str:
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    hostname = parsed.hostname
    if hostname is None:
        raise ValueError("host must include a hostname")
    rendered_host = f"[{hostname}]" if ":" in hostname else hostname
    return f"{rendered_host}:{port}"


def normalize_url(host: str = DEFAULT_BYSTRO_URL, port: int | None = None) -> str:
    """Normalize a Bystro base URL without adding default ports."""

    raw = host.strip()
    if not raw:
        raise ValueError("host cannot be empty")
    if "://" not in raw:
        raw = _url_with_inferred_scheme(raw)

    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("host protocol must be http or https")
    if parsed.hostname is None:
        raise ValueError("host must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("host must not include credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("host must not include a query string or fragment")

    try:
        parsed_port = parsed.port
    except ValueError as exc:
        raise ValueError("host contains an invalid port") from exc
    netloc = _replace_port(parsed, port) if port is not None else parsed.netloc
    if port is None and parsed_port is None:
        netloc = parsed.netloc
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, netloc, path, "", ""))


def _credentials_directory() -> Path:
    return Path(DEFAULT_DIR)


def _credentials_path() -> Path:
    return Path(CREDENTIALS_PATH)


def _harden_credentials_permissions(directory: Path, destination: Path) -> None:
    os.chmod(directory, 0o700)
    os.chmod(destination, 0o600)


def load_state() -> CachedAuth | None:
    """Load the cached dashboard credential, if one exists."""

    directory = _credentials_directory()
    destination = _credentials_path()
    try:
        _harden_credentials_permissions(directory, destination)
        encoded = destination.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise AuthenticationError("Could not read cached Bystro authentication state") from exc
    if not encoded:
        return None
    try:
        return mjson.decode(encoded, type=CachedAuth)
    except (DecodeError, ValidationError) as exc:
        raise AuthenticationError("Cached Bystro authentication state is invalid") from exc


def save_state(data: CachedAuth, print_result: bool = False) -> None:
    """Atomically save an authentication token with user-only permissions."""

    directory = _credentials_directory()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    destination = _credentials_path()
    encoded = mjson.encode(data)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{STATE_FILE}.", dir=directory)
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        _harden_credentials_permissions(directory, destination)
    except Exception:
        # This is the cleanup boundary for an incomplete atomic write.
        logger.exception("Could not atomically save Bystro authentication state")
        with suppress(OSError):
            os.close(fd)
        with suppress(FileNotFoundError):
            temporary_path.unlink()
        raise

    if print_result:
        display_data = {
            "email": data.email,
            "access_token": "[redacted]",
            "url": data.url,
        }
        formatted = mjson.format(mjson.encode(display_data), indent=4).decode("utf-8")
        print(f"\nSaved auth credentials to {destination}:\n{formatted}")


def _response_payload(response: requests.Response) -> dict[str, object]:
    try:
        payload = cast(object, response.json())
    except (ValueError, TypeError) as exc:
        raise AuthenticationError(
            "Bystro returned an invalid authentication response",
            status_code=response.status_code,
        ) from exc
    if not isinstance(payload, dict):
        raise AuthenticationError(
            "Bystro returned an invalid authentication response",
            status_code=response.status_code,
        )
    raw_payload = cast(dict[object, object], payload)
    return {key: value for key, value in raw_payload.items() if isinstance(key, str)}


def _auth_error(response: requests.Response, action: str) -> AuthenticationError:
    try:
        payload = _response_payload(response)
    except AuthenticationError:
        payload = {}
    raw_code = payload.get("error")
    code = raw_code if isinstance(raw_code, str) and raw_code.strip() else None
    raw_missing = payload.get("missing")
    missing = (
        tuple(item for item in raw_missing if isinstance(item, str))
        if isinstance(raw_missing, list)
        else ()
    )
    error_type = (
        LegalConsentRequiredError
        if code in {"consent_required", "legal_consent_required", "over18_required"}
        else AuthenticationError
    )
    return error_type(
        f"Bystro {action} failed with status {response.status_code} ({response.reason})",
        status_code=response.status_code,
        code=code,
        missing=missing,
    )


def _request_token(
    *,
    session: requests.Session,
    method: Literal["PUT", "POST"],
    url: str,
    data: dict[str, str],
    action: str,
    timeout: float,
) -> str:
    try:
        if method == "PUT":
            response = session.put(
                url,
                data=data,
                timeout=timeout,
                allow_redirects=False,
            )
        else:
            response = session.post(
                url,
                data=data,
                timeout=timeout,
                allow_redirects=False,
            )
    except requests.RequestException as exc:
        logger.debug("Bystro %s request failed", action, exc_info=True)
        raise AuthenticationError(f"Could not reach Bystro to {action}") from exc
    if response.status_code != 200:
        raise _auth_error(response, action)
    payload = _response_payload(response)
    token = payload.get("bystro_access_token")
    if not isinstance(token, str) or not token.strip():
        raise AuthenticationError(
            "Bystro returned an authentication response without a token",
            status_code=response.status_code,
        )
    return token.strip()


def _authenticate_site_gate(
    session: requests.Session,
    *,
    host: str,
    access_code: str,
    timeout: float,
) -> None:
    normalized_code = access_code.strip()
    if not normalized_code:
        raise ValueError("site_access_code cannot be empty")
    try:
        response = session.post(
            f"{host}/api/site-gate/authenticate",
            json={"password": normalized_code},
            timeout=timeout,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        logger.debug("Bystro site-gate authentication request failed", exc_info=True)
        raise SiteGateAuthenticationError("Could not reach Bystro to authenticate site access") from exc
    if response.status_code != 200:
        raise SiteGateAuthenticationError(
            "Bystro site-access authentication was rejected",
            status_code=response.status_code,
            code="site_access_rejected",
        )


def signup(
    email: str,
    password: str,
    name: str,
    host: str = DEFAULT_BYSTRO_URL,
    port: int | None = None,
    print_result: bool = False,
    cache: bool = True,
    *,
    legal_consent: LegalConsent | None = None,
    site_access_code: str | None = None,
    timeout: float = _REQUEST_TIMEOUT_SECONDS,
) -> CachedAuth:
    """Create a dashboard account and cache its browser JWT."""

    fq_host = normalize_url(host, port)
    if print_result:
        print(f"\nSigning up for Bystro with email: {email}, name: {name}")
    data = {"email": email, "name": name, "password": password}
    if legal_consent is not None:
        data.update(legal_consent.form_fields())
    with requests.Session() as session:
        if site_access_code is not None:
            _authenticate_site_gate(
                session,
                host=fq_host,
                access_code=site_access_code,
                timeout=timeout,
            )
        token = _request_token(
            session=session,
            method="PUT",
            url=f"{fq_host}/api/user",
            data=data,
            action="signup",
            timeout=timeout,
        )
    state = CachedAuth(access_token=token, url=fq_host, email=email)
    if cache:
        save_state(state, print_result)
    if print_result:
        print("\nSignup and authentication successful.\n")
    return state


def login(
    email: str,
    password: str,
    host: str = DEFAULT_BYSTRO_URL,
    port: int | None = None,
    print_result: bool = False,
    cache: bool = True,
    *,
    legal_consent: LegalConsent | None = None,
    site_access_code: str | None = None,
    timeout: float = _REQUEST_TIMEOUT_SECONDS,
) -> CachedAuth:
    """Log in through the dashboard and optionally cache its browser JWT."""

    fq_host = normalize_url(host, port)
    if print_result:
        print(f"\nLogging into {fq_host} with email: {email}.")
    data = {"email": email, "password": password}
    if legal_consent is not None:
        data.update(legal_consent.form_fields())
    with requests.Session() as session:
        if site_access_code is not None:
            _authenticate_site_gate(
                session,
                host=fq_host,
                access_code=site_access_code,
                timeout=timeout,
            )
        token = _request_token(
            session=session,
            method="POST",
            url=f"{fq_host}/api/user/auth/local",
            data=data,
            action="login",
            timeout=timeout,
        )
    state = CachedAuth(access_token=token, url=fq_host, email=email)
    if cache:
        save_state(state, print_result)
    if print_result:
        print("\nLogin successful. You may now use the Bystro API.\n")
    return state


def authenticate(cached_auth: CachedAuth | None = None) -> tuple[CachedAuth, dict[str, str]]:
    """Return an auth state and its standard Bearer header."""

    state = cached_auth if cached_auth is not None else load_state()
    if state is None:
        raise AuthenticationError("You are not logged in. Call bystro.api.auth.login first.")
    return state, {"Authorization": f"Bearer {state.access_token}"}


def get_user(
    print_result: bool = False,
    cached_auth: CachedAuth | None = None,
    *,
    timeout: float = _REQUEST_TIMEOUT_SECONDS,
) -> UserProfile:
    """Fetch the current dashboard user profile."""

    state, auth_header = authenticate(cached_auth)
    try:
        response = requests.get(
            f"{state.url}/api/user/me",
            headers=auth_header,
            timeout=timeout,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        logger.debug("Bystro profile request failed", exc_info=True)
        raise AuthenticationError("Could not reach Bystro to fetch the user profile") from exc
    if response.status_code != 200:
        raise _auth_error(response, "profile request")
    payload = _response_payload(response)
    try:
        profile = mjson.decode(mjson.encode(payload), type=UserProfile)
    except (DecodeError, ValidationError) as exc:
        raise AuthenticationError(
            "Bystro returned an invalid user profile",
            status_code=response.status_code,
        ) from exc
    if print_result:
        print(f"\nFetched profile for {profile.email}.\n")
    return profile


def logout(print_result: bool = False) -> None:
    """Remove the locally cached dashboard credential."""

    with suppress(FileNotFoundError):
        _credentials_path().unlink()
    if print_result:
        print(f"\nLogged out. Removed auth credentials from {_credentials_path()}\n")


__all__ = [
    "AuthenticationError",
    "CREDENTIALS_PATH",
    "CachedAuth",
    "DEFAULT_BYSTRO_URL",
    "LegalConsentRequiredError",
    "LegalConsent",
    "LoginResponse",
    "SiteGateAuthenticationError",
    "SignupResponse",
    "UserProfile",
    "authenticate",
    "get_user",
    "load_state",
    "login",
    "logout",
    "normalize_url",
    "save_state",
    "signup",
]
