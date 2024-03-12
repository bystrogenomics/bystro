from bystro.cli.cli import (
    signup_cli,
    login_cli,
    get_jobs_cli,
    create_jobs_cli,
    get_user_cli,
)


def test_signup_cli(mocker):
    mocker.patch("bystro.cli.cli.signup", return_value="signup_result")

    args = mocker.Mock(
        email="test@example.com", password="password123", name="Test User", host="host", port=443
    )

    result = signup_cli(args)

    assert result == "signup_result"


def test_login_cli(mocker):
    mocker.patch("bystro.cli.cli.login", return_value="login_result")

    args = mocker.Mock(email="test@example.com", password="password123", host="host", port=443)

    result = login_cli(args)

    assert result == "login_result"


def test_get_jobs_cli(mocker):
    mocker.patch("bystro.cli.cli.get_jobs", return_value="jobs_result")

    args = mocker.Mock(type="completed", id=None)

    result = get_jobs_cli(args)

    assert result == "jobs_result"


def test_create_jobs_cli(mocker):
    mocker.patch("bystro.cli.cli.create_jobs", return_value="create_jobs_result")

    args = mocker.Mock(
        files=["file1.vcf", "file2.vcf"], names=["job1", "job2"], assembly="hg38", create_index=True
    )

    result = create_jobs_cli(args)

    assert result == "create_jobs_result"


def test_get_user_cli(mocker):
    mocker.patch("bystro.cli.cli.get_user", return_value="user_profile")

    result = get_user_cli()

    assert result == "user_profile"
