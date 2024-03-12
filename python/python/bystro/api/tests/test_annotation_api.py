import os

from msgspec import json
import pytest

from bystro.api.annotation import (
    JOB_TYPE_ROUTE_MAP,
    get_jobs,
    create_jobs,
)

from bystro.api.auth import (
    CachedAuth,
)


EXAMPLE_CACHED_AUTH = CachedAuth(email="blah", url="http://localhost", access_token="blah")

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


def test_create_job(mocker):
    mocker.patch(
        "bystro.api.annotation.authenticate",
        return_value=(
            EXAMPLE_CACHED_AUTH,
            "localhost:8080",
        ),
    )
    mocker.patch(
        "requests.post",
        return_value=mocker.Mock(status_code=200, json=lambda: {"success": True}),
    )

    files = [os.path.join(os.path.dirname(__file__), "trio.trim.vep.short.vcf.gz")]
    assembly = "hg38"
    index = True
    names = None

    response = create_jobs(files, assembly, names, index, print_result=False)
    assert response == [{"success": True}]


def test_get_job_fail_validation(mocker):
    mocker.patch(
        "bystro.api.annotation.authenticate",
        return_value=(EXAMPLE_CACHED_AUTH, {"Authorization": "Bearer TOKEN"}),
    )

    with pytest.raises(ValueError, match="Please specify either a job id or a job type"):
        job_type = None
        job_id = None

        get_jobs(job_type, job_id, print_result=False)

    with pytest.raises(ValueError, match="Please specify either a job id or a job type, not both"):
        job_type = "completed"
        job_id = "1234"

        get_jobs(job_type, job_id, print_result=False)

    with pytest.raises(
        ValueError,
        match=f"Invalid job type: dasfa. Valid types are: {', '.join(JOB_TYPE_ROUTE_MAP.keys())}",
    ):

        job_type = "dasfa"
        job_id = None
        get_jobs(job_type, job_id, print_result=False)


def test_get_job_list(mocker):
    mocker.patch(
        "bystro.api.annotation.authenticate",
        return_value=(EXAMPLE_CACHED_AUTH, {"Authorization": "Bearer TOKEN"}),
    )
    mocker.patch(
        "requests.get",
        return_value=mocker.Mock(status_code=200, text="[]"),  # noqa: PIE807
    )

    job_type = "completed"
    job_id = None

    response = get_jobs(job_type, job_id, print_result=False)
    assert response == []

    job_type = "failed"
    job_id = None
    response = get_jobs(job_type, job_id, print_result=False)
    assert response == []

    job_type = "public"
    job_id = None
    response = get_jobs(job_type, job_id, print_result=False)
    assert response == []

    job_type = "shared"
    job_id = None
    response = get_jobs(job_type, job_id, print_result=False)
    assert response == []

    job_type = "all"
    job_id = None
    response = get_jobs(job_type, job_id, print_result=False)
    assert response == []

    job_type = "incomplete"
    job_id = None
    response = get_jobs(job_type, job_id, print_result=False)
    assert response == []


def test_get_job(mocker):
    mocker.patch(
        "bystro.api.annotation.authenticate",
        return_value=(EXAMPLE_CACHED_AUTH, {"Authorization": "Bearer TOKEN"}),
    )

    mocker.patch(
        "requests.get",
        return_value=mocker.Mock(status_code=200, text=json.encode(EXAMPLE_JOB).decode("utf-8")),
    )

    job_id = "12341"
    job_type = None
    response = get_jobs(job_type, job_id, print_result=False)

    parsed_job = EXAMPLE_JOB.copy()

    assert response == parsed_job
