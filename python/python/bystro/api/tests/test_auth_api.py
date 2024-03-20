from datetime import datetime, timezone

from msgspec import json
import pytest

from bystro.api.auth import (
    CachedAuth,
    SignupResponse,
    UserProfile,
    load_state,
    save_state,
    signup,
    login,
    _fq_host,
    get_user,
    CREDENTIALS_PATH,
)

EXAMPLE_DATE_STRING = "2023-09-06T05:45:01.446Z"

EXAMPLE_FORMAT_STRING = "%Y-%m-%dT%H:%M:%S.%fZ"

EXAMPLE_USER = UserProfile(
    options={"autoUploadToS3": False},
    _id="93902394u2903902",
    name="Test Account 3",
    email="test@gmail.com",
    accounts=["bystro"],
    role="user",
    last_login=datetime.strptime(EXAMPLE_DATE_STRING, EXAMPLE_FORMAT_STRING).replace(
        tzinfo=timezone.utc
    ),
)

EXAMPLE_SIGNUP_RESPONSE = SignupResponse(access_token="20302493029=02934")

EXAMPLE_CACHED_AUTH = CachedAuth(email="blah", url="http://localhost", access_token="blah")


def test_load_state_existing_file(mocker):
    mocker.patch("os.path.exists", return_value=True)

    mocker.patch(
        "builtins.open",
        mocker.mock_open(read_data=json.encode(EXAMPLE_CACHED_AUTH).decode("utf-8")),
    )
    result = load_state()

    assert result == EXAMPLE_CACHED_AUTH


def test_load_state_no_file(mocker):
    mocker.patch("os.path.exists", return_value=False)
    result = load_state()
    assert result is None


def test_save_state(mocker):
    mock_open = mocker.patch("builtins.open", mocker.mock_open())

    save_state(EXAMPLE_CACHED_AUTH, print_result=False)
    mock_open.assert_called_once_with(CREDENTIALS_PATH, "w", encoding="utf-8")


@pytest.mark.parametrize(
    "status_code, exception_message",
    [
        (404, "Login failed with response status: 404. Error: \nerror\n"),
        (500, "Login failed with response status: 500. Error: \nserver error\n"),
    ],
)
def test_login_failure(mocker, status_code, exception_message):
    mocker.patch(
        "requests.post",
        return_value=mocker.Mock(
            status_code=status_code,
            text="error" if status_code == 404 else "server error",
        ),
    )
    host = "localhost"
    port = 8080
    email = "test@example.com"
    password = "password"
    with pytest.raises(RuntimeError, match=exception_message):
        login(email, password, host, port, print_result=False)


def test_signup(mocker):
    expected_response = EXAMPLE_SIGNUP_RESPONSE
    mocker.patch(
        "bystro.api.auth.save_state",
        return_value=(EXAMPLE_CACHED_AUTH, {"Authorization": "Bearer TOKEN"}),
    )
    mocker.patch(
        "requests.put",
        return_value=mocker.Mock(status_code=200, text=json.encode(expected_response).decode("utf-8")),
    )
    email = "test@example.com"
    host = "http://localhost"
    port = 8080
    password = "password"
    name = "test"

    response = signup(email, password, name, host, port, print_result=False)

    url = _fq_host(host, port)

    expected_return = CachedAuth(email=email, url=url, access_token=expected_response.access_token)
    assert response == expected_return


def test_get_user(mocker):
    mocker.patch(
        "bystro.api.auth.authenticate",
        return_value=(EXAMPLE_CACHED_AUTH, {"Authorization": "Bearer TOKEN"}),
    )
    mocker.patch(
        "requests.get",
        return_value=mocker.Mock(status_code=200, text=json.encode(EXAMPLE_USER).decode("utf-8")),
    )

    user = get_user(print_result=False)
    assert user == EXAMPLE_USER
