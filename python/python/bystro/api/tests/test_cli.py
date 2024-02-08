from argparse import Namespace
from datetime import datetime, timezone
import os

from msgspec import json
import pytest

from bystro.api.cli import (
    CachedAuth,
    UserProfile,
    SignupResponse,
    load_state,
    save_state,
    signup,
    login,
    JOB_TYPE_ROUTE_MAP,
    get_jobs,
    create_job,
    get_user,
    _fq_host,
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
    lastLogin=datetime.strptime(EXAMPLE_DATE_STRING, EXAMPLE_FORMAT_STRING).replace(
        tzinfo=timezone.utc
    ),
)

EXAMPLE_SIGNUP_RESPONSE = SignupResponse(access_token="20302493029=02934")

EXAMPLE_CACHED_AUTH = CachedAuth(
    email="blah", url="http://localhost", access_token="blah"
)

EXAMPLE_JOB = {
    "_id": "64db4e67fb86b79cbda4f386",
    "assembly": "hg38",
    "options": {"index": True},
    "inputQueryConfig": {"fieldNames": []},
    "exports": {"archivedSubmissions": []},
    "search": {
        "fieldNames": [],
        "indexConfig": [],
        "archivedSubmissions": [],
        "queries": [],
    },
    "type": "annotation",
    "visibility": "private",
    "ancestry": {},
    "archivedSubmissions": [],
    "actionsTaken": [],
    "expireDate": "2023-08-22T10:07:35.865Z",
    "userID": "64d6264274e4fab6baa0d198",
    "inputFileName": "trio.trim.vep.vcf.gz",
    "name": "trio.trim.vep.vcf",
    "outputBaseFileName": "trio_trim_vep_vcf",
    "submission": {
        "state": "started",
        "attempts": 1,
        "log": {
            "progress": 0,
            "skipped": 0,
            "messages": [
                "Job Submitted!",
            ],
        },
        "addedFileNames": [],
        "_id": "64db4e68fb86b79cbda4f387",
        "type": "annotation",
        "submittedDate": "2023-08-15T10:07:36.027Z",
        "queueId": "1538",
        "startedDate": "2023-08-15T10:07:37.045Z",
    },
    "config": json.encode(
        {
            "database_dir": "hidden",
            "files_dir": "hidden",
            "chromosomes": [
                "chr1",
                "chr2",
            ],
            "assembly": "hg38",
        }
    ).decode("utf-8"),
}


def test_load_state_existing_file(mocker):
    mocker.patch("os.path.exists", return_value=True)

    mocker.patch(
        "builtins.open",
        mocker.mock_open(
            read_data=json.encode(EXAMPLE_CACHED_AUTH).decode("utf-8")
        ),
    )
    result = load_state("./")

    assert result == EXAMPLE_CACHED_AUTH


def test_load_state_no_file(mocker):
    mocker.patch("os.path.exists", return_value=False)
    result = load_state("./")
    assert result is None


def test_save_state(mocker):
    mock_open = mocker.patch("builtins.open", mocker.mock_open())

    save_state(EXAMPLE_CACHED_AUTH, "./", print_result=False)
    mock_open.assert_called_once_with(
        "./bystro_authentication_token.json", "w", encoding="utf-8"
    )


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
    args = Namespace(
        host="localhost",
        port=8080,
        email="test@example.com",
        password="password",
        dir="./",
    )
    with pytest.raises(RuntimeError, match=exception_message):
        login(args, print_result=False)


def test_create_job(mocker):
    mocker.patch(
        "bystro.api.cli.authenticate",
        return_value=(
            EXAMPLE_CACHED_AUTH,
            "localhost:8080",
        ),
    )
    mocker.patch(
        "requests.post",
        return_value=mocker.Mock(status_code=200, json=lambda: {"success": True}),
    )
    args = Namespace(
        files=[os.path.join(os.path.dirname(__file__), "trio.trim.vep.short.vcf.gz")],
        assembly="hg38",
        index=True,
        dir="./",
    )
    response = create_job(args, print_result=False)
    assert response == {"success": True}


def test_get_job_fail_validation(mocker):
    mocker.patch(
        "bystro.api.cli.authenticate",
        return_value=(EXAMPLE_CACHED_AUTH, {"Authorization": "Bearer TOKEN"}),
    )

    with pytest.raises(
        ValueError, match="Please specify either a job id or a job type"
    ):
        args = Namespace(dir="./", type=None, id=None)
        get_jobs(args, print_result=False)

    with pytest.raises(
        ValueError, match="Please specify either a job id or a job type, not both"
    ):
        args = Namespace(dir="./", type="completed", id="1234")
        get_jobs(args, print_result=False)

    with pytest.raises(
        ValueError,
        match=f"Invalid job type: dasfa. Valid types are: {', '.join(JOB_TYPE_ROUTE_MAP.keys())}",
    ):
        args = Namespace(dir="./", type="dasfa", id=None)
        get_jobs(args, print_result=False)


def test_get_job_list(mocker):
    mocker.patch(
        "bystro.api.cli.authenticate",
        return_value=(EXAMPLE_CACHED_AUTH, {"Authorization": "Bearer TOKEN"}),
    )
    mocker.patch(
        "requests.get",
        return_value=mocker.Mock(status_code=200, text="[]"),  # noqa: PIE807
    )

    args = Namespace(dir="./", type="completed", id=None)

    response = get_jobs(args, print_result=False)
    assert response == []

    args = Namespace(dir="./", type="failed", id=None)
    response = get_jobs(args, print_result=False)
    assert response == []

    args = Namespace(dir="./", type="public", id=None)
    response = get_jobs(args, print_result=False)
    assert response == []

    args = Namespace(dir="./", type="shared", id=None)
    response = get_jobs(args, print_result=False)
    assert response == []

    args = Namespace(dir="./", type="all", id=None)
    response = get_jobs(args, print_result=False)
    assert response == []

    args = Namespace(dir="./", type="incomplete", id=None)
    response = get_jobs(args, print_result=False)
    assert response == []


def test_get_job(mocker):
    mocker.patch(
        "bystro.api.cli.authenticate",
        return_value=(EXAMPLE_CACHED_AUTH, {"Authorization": "Bearer TOKEN"}),
    )

    mocker.patch(
        "requests.get",
        return_value=mocker.Mock(
            status_code=200, text=json.encode(EXAMPLE_JOB).decode("utf-8")
        ),
    )

    args = Namespace(dir="./", id="12341", type=None)
    response = get_jobs(args, print_result=False)

    parsed_job = EXAMPLE_JOB.copy()
    parsed_job["config"] = json.decode(parsed_job["config"]) # type: ignore

    assert response == parsed_job


def test_signup(mocker):
    expected_response = EXAMPLE_SIGNUP_RESPONSE
    mocker.patch(
        "bystro.api.cli.save_state",
        return_value=(EXAMPLE_CACHED_AUTH, {"Authorization": "Bearer TOKEN"}),
    )
    mocker.patch(
        "requests.put",
        return_value=mocker.Mock(
            status_code=200, text=json.encode(expected_response).decode("utf-8")
        ),
    )
    email = "test@example.com"
    host = "http://localhost"
    port = 8080
    args = Namespace(
        dir="./", email=email, password="password", name="test", host=host, port=port
    )

    response = signup(args, print_result=False)

    url = _fq_host(Namespace(host=host, port=port))

    expected_return = CachedAuth(
        email=email, url=url, access_token=expected_response.access_token
    )
    assert response == expected_return


def test_get_user(mocker):
    mocker.patch(
        "bystro.api.cli.authenticate",
        return_value=(EXAMPLE_CACHED_AUTH, {"Authorization": "Bearer TOKEN"}),
    )
    mocker.patch(
        "requests.get",
        return_value=mocker.Mock(
            status_code=200, text=json.encode(EXAMPLE_USER).decode("utf-8")
        ),
    )
    args = Namespace(dir="./", email="test@example.com", password="password")
    user = get_user(args, print_result=False)
    assert user == EXAMPLE_USER
