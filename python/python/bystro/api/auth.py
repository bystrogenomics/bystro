import os

import requests
import datetime

from msgspec import Struct, json as mjson

DEFAULT_DIR = os.path.expanduser("~/.bystro")
STATE_FILE = "bystro_authentication_token.json"

CREDENTIALS_PATH = os.path.join(DEFAULT_DIR, STATE_FILE)

if not os.path.exists(DEFAULT_DIR):
    os.makedirs(DEFAULT_DIR, exist_ok=True)
    print(f"Created default credentials path: {CREDENTIALS_PATH}")


class SignupResponse(Struct):
    """
    The response body for signing up for Bystro.

    Attributes
    ----------
    access_token : str
        The access token, which authorizes further API requests
    """

    access_token: str


class LoginResponse(Struct):
    """
    The response body for logging in to Bystro.

    Attributes
    ----------
    access_token : str
        The access token, which authorizes further API requests
    """

    access_token: str


class CachedAuth(Struct):
    """
    The authentication state.

    Attributes
    ----------
    email : str
        The email of the user.
    access_token : str
        The access token, which authorizes further API requests
    url : str
        The url of the Bystro server.
    """

    email: str
    access_token: str
    url: str


class UserProfile(Struct, rename="camel"):
    """
    The response body for fetching the user profile.

    Attributes
    ----------
    options : dict
        The user options.
    _id : str
        The id of the user.
    name : str
        The name of the user.
    email : str
        The email of the user.
    accounts : list[str]
        The accounts of the user.
    role : str
        The role of the user.
    lastLogin : datetime.datetime
        The date the user last logged in.
    """

    _id: str
    options: dict
    name: str
    email: str
    accounts: list[str]
    role: str
    last_login: datetime.datetime


def _fq_host(host: str, port: int | None = None) -> str:
    """
    Returns the fully qualified host, e.g. https://bystro-dev.emory.edu:443

    Parameters
    ----------
    host : str
        The hostname or IP address of the server.
    port : int | None
        (Optional) The port number on which the server is listening.

    Returns
    -------
    str
        The fully qualified host.
    """
    if port is None:
        # parse the port from the host protocol
        if host.startswith("https"):
            port = 443
        elif host.startswith("http"):
            port = 80
        else:
            raise ValueError(f"Invalid host protocol: {host}")

    return f"{host}:{port}"


def load_state() -> CachedAuth | None:
    """
    Loads the authentication state from the state directory.

    Returns
    -------
    CachedAuth | None
        The authentication state, or None if the state file doesn't exist.
    """
    if os.path.exists(CREDENTIALS_PATH):
        with open(CREDENTIALS_PATH, "r", encoding="utf-8") as f:
            data = f.read()

            if not data:
                return None

            return mjson.decode(data, type=CachedAuth)

    return None


def save_state(data: CachedAuth, print_result=True) -> None:
    """
    Saves the authentication state to a file.

    Parameters
    ----------
    data : CachedAuth
        The data to save.
    print_result : bool, optional
        Whether to print the result of the save operation, by default True.

    Returns
    --------
    None
    """
    encoded_data = mjson.encode(data).decode("utf-8")

    with open(CREDENTIALS_PATH, "w", encoding="utf-8") as f:
        f.write(encoded_data)

    if print_result:
        print(f"\nSaved auth credentials to {CREDENTIALS_PATH}:\n{mjson.format(encoded_data, indent=4)}")


def signup(
    email: str, password: str, name: str, host: str, port: int | None = None, print_result=True
) -> CachedAuth:
    """
    Signs up for Bystro with the given email, name, and password. Additionally, logs in and
    saves the credentials, to enable API calls without re-authenticating.

    Parameters
    ----------
    email : str
        The email address for the account.
    name : str
        The name of the user.
    password : str
        The password for the account.
    host : str
        The hostname or IP address of the Bystro server.
    port : int | None
        (Optional) The port number on which the Bystro server is listening.
    print_result : bool, optional
        Whether to print the result of the signup operation, by default True.

    Returns
    -------
    CachedAuth
        The cached authentication state.
    """
    if os.path.exists(CREDENTIALS_PATH):
        print("Existing session found, logging out")
        logout()

    if print_result:
        print(f"\nSigning up for Bystro with email: {email}, name: {name}")

    fq_host = _fq_host(host, port)
    url = f"{fq_host}/api/user"

    data = {"email": email, "name": name, "password": password}

    response = requests.put(url, data=data, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"Login failed with response status: {response.status_code}. Error: \n{response.text}\n"
        )

    res = mjson.decode(response.text, type=SignupResponse)
    state = CachedAuth(
        access_token=res.access_token,
        url=fq_host,
        email=email,
    )

    save_state(
        state,
        print_result,
    )

    if print_result:
        print("\nSignup & authentication successful. You may now use the Bystro API!\n")

    return state


def login( email: str, password: str, host: str, port: int | None = None, print_result=True,
    ) -> CachedAuth:
    """
    Logs in to the server with the provided credentials and saves the authentication state to a file.

    Parameters
    ----------
    email : str
        The email address used for login.
    password : str
        The password for the account.
    host : str
        The hostname or IP address of the Bystro server.
    port : int | None
        (Optional) The port number on which the Bystro server is listening.
    print_result : bool, optional
        Whether to print the result of the login operation, by default True.

    Returns
    -------
    CachedAuth
        The cached authentication state.
    """
    if os.path.exists(CREDENTIALS_PATH):
        print("Existing session found, logging out")
        logout()

    fq_host = _fq_host(host, port)

    if print_result:
        print(f"\nLogging into {fq_host} with email: {email}.")

    url = f"{fq_host}/api/user/auth/local"

    body = {"email": email, "password": password}

    response = requests.post(url, data=body, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"Login failed with response status: {response.status_code}. Error: \n{response.text}\n"
        )

    if print_result:
        print("response.text", response.text)

    res = mjson.decode(response.text, type=LoginResponse)
    state = CachedAuth(access_token=res.access_token, url=fq_host, email=email)
    save_state(state, print_result)

    if print_result:
        print("\nLogin successful. You may now use the Bystro API!\n")

    return state


def authenticate() -> tuple[CachedAuth, dict]:
    """
    Authenticates the user and returns the url, auth header, and email.


    Returns
    -------
    tuple[CachedAuth, dict]
        The cached auth credentials and auth header
    """
    state = load_state()

    if not state:
        raise ValueError("\n\nYou are not logged in. Please login first.\n")

    header = {"Authorization": f"Bearer {state.access_token}"}
    return state, header


def get_user(print_result=True) -> UserProfile:
    """
    Fetches the user profile.

    Parameters
    ----------
    print_result : bool, optional
        Whether to print the result of the user profile fetch operation, by default True.

    Returns
    -------
    UserProfile
        The user profile
    """
    if print_result:
        print("\n\nFetching user profile\n")

    state, auth_header = authenticate()

    response = requests.get(state.url + "/api/user/me", headers=auth_header, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"Fetching profile failed with response status: {response.status_code}.\
                Error: \n{response.text}\n"
        )

    user_profile = mjson.decode(response.text, type=UserProfile)

    if print_result:
        print(f"\nFetched Profile for email {state.email}\n")
        print(mjson.format(response.text, indent=4))
        print("\n")

    return user_profile


def logout(print_result=True) -> None:
    """
    Logs out of the Bystro server by deleting the authentication state file.

    Parameters
    ----------
    print_result : bool, optional
        Whether to print the result of the logout operation, by default True.

    Returns
    -------
    None
    """
    if os.path.exists(CREDENTIALS_PATH):
        os.remove(CREDENTIALS_PATH)

    if print_result:
        print(f"\nLogged out. Removed auth credentials from {CREDENTIALS_PATH}\n")
