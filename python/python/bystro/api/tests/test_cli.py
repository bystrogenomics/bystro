import pytest
import os

from bystro.api.cli import (
    load_state,
    save_state,
    signup,
    login,
    JOB_TYPE_ROUTE_MAP,
    get_jobs,
    create_job,
    get_user,
)

from types import SimpleNamespace
import json

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
        "queueID": "1538",
        "startedDate": "2023-08-15T10:07:37.045Z",
    },
    "config": json.dumps(
        {
            "database_dir": "hidden",
            "files_dir": "hidden",
            "chromosomes": [
                "chr1",
                "chr2",
            ],
            "assembly": "hg38",
        }
    ),
}


def test_load_state_existing_file(mocker):
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("builtins.open", mocker.mock_open(read_data='{"key": "value"}'))
    result = load_state("./")
    assert result == {"key": "value"}


def test_load_state_no_file(mocker):
    mocker.patch("os.path.exists", return_value=False)
    result = load_state("./")
    assert result == {}


def test_save_state(mocker):
    mock_open = mocker.patch("builtins.open", mocker.mock_open())
    save_state({"key": "value"}, "./", print_result=False)
    mock_open.assert_called_once_with("./bystro_authentication_token.json", "w")


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
    args = SimpleNamespace(
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
            "localhost:8080",
            {"Authorization": "Bearer TOKEN"},
            "test@example.com",
        ),
    )
    mocker.patch(
        "requests.post",
        return_value=mocker.Mock(status_code=200, json=lambda: {"success": True}),
    )
    args = SimpleNamespace(
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
        return_value=(
            "localhost:8080",
            {"Authorization": "Bearer TOKEN"},
            "test@example.com",
        ),
    )

    with pytest.raises(
        ValueError, match="Please specify either a job id or a job type"
    ):
        args = SimpleNamespace(dir="./", type=None, id=None)
        get_jobs(args, print_result=False)

    with pytest.raises(
        ValueError, match="Please specify either a job id or a job type, not both"
    ):
        args = SimpleNamespace(dir="./", type="completed", id="1234")
        get_jobs(args, print_result=False)

    with pytest.raises(
        ValueError,
        match=f"Invalid job type: dasfa. Valid types are: {','.join(JOB_TYPE_ROUTE_MAP.keys())}",
    ):
        args = SimpleNamespace(dir="./", type="dasfa", id=None)
        get_jobs(args, print_result=False)


def test_get_job_list(mocker):
    mocker.patch(
        "bystro.api.cli.authenticate",
        return_value=(
            "localhost:8080",
            {"Authorization": "Bearer TOKEN"},
            "test@example.com",
        ),
    )
    mocker.patch(
        "requests.get", return_value=mocker.Mock(status_code=200, json=lambda: []) # noqa: PIE807
    )

    args = SimpleNamespace(dir="./", type="completed", id=None)
    print("args", args)
    response = get_jobs(args, print_result=False)
    assert response == []

    args = SimpleNamespace(dir="./", type="failed", id=None)
    response = get_jobs(args, print_result=False)
    assert response == []

    args = SimpleNamespace(dir="./", type="public", id=None)
    response = get_jobs(args, print_result=False)
    assert response == []

    args = SimpleNamespace(dir="./", type="shared", id=None)
    response = get_jobs(args, print_result=False)
    assert response == []

    args = SimpleNamespace(dir="./", type="all", id=None)
    response = get_jobs(args, print_result=False)
    assert response == []

    args = SimpleNamespace(dir="./", type="incomplete", id=None)
    response = get_jobs(args, print_result=False)
    assert response == []


def test_get_job(mocker):
    mocker.patch(
        "bystro.api.cli.authenticate",
        return_value=(
            "localhost:8080",
            {"Authorization": "Bearer TOKEN"},
            "test@example.com",
        ),
    )

    mocker.patch(
        "requests.get",
        return_value=mocker.Mock(status_code=200, json=lambda: EXAMPLE_JOB.copy()),
    )

    args = SimpleNamespace(dir="./", id="12341", type=None)
    response = get_jobs(args, print_result=False)

    parsed_job = EXAMPLE_JOB.copy()
    parsed_job["config"] = json.loads(parsed_job["config"])

    assert response == parsed_job


def test_signup(mocker):
    expected_response = {"access_token": "TOKEN"}
    mocker.patch(
        "bystro.api.cli.save_state",
        return_value=(
            "access_token",
            {"Authorization": "Bearer TOKEN"},
            "test@example.com",
        ),
    )
    mocker.patch(
        "requests.put",
        return_value=mocker.Mock(status_code=200, json=lambda: expected_response),
    )
    args = SimpleNamespace(
        dir="./",
        email="test@example.com",
        password="password",
        name="test",
        host="localhost",
        port=8080,
    )
    response = signup(args, print_result=False)
    assert response == expected_response


def test_get_user(mocker):
    mocker.patch(
        "bystro.api.cli.authenticate",
        return_value=(
            "localhost:8080",
            {"Authorization": "Bearer TOKEN"},
            "test@example.com",
        ),
    )
    mocker.patch(
        "requests.request",
        return_value=mocker.Mock(
            status_code=200, json=lambda: {"email": "test@example.com"}
        ),
    )
    args = SimpleNamespace(dir="./", email="test@example.com", password="password")
    user = get_user(args, print_result=False)
    assert user == {"email": "test@example.com"}
