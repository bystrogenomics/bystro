"""Test ancestry listener."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from bystro.ancestry import listener  # noqa: F401
from bystro.ancestry.ancestry_types import AncestrySubmission
from bystro.ancestry.listener import (
    AncestryJobData,
    AncestryModel,
    _infer_ancestry,
    completed_msg_fn,
    handler_fn_factory,
    submit_msg_fn,
)
from bystro.ancestry.train import POPS
from bystro.beanstalkd.messages import ProgressMessage
from bystro.beanstalkd.worker import ProgressPublisher

SAMPLES = [f"sample{i}" for i in range(len(POPS))]
VARIANTS = ["variant1", "variant2", "variant3"]
PC_COLUMNS = ["pc1", "pc2", "pc3", "pc4"]
TRAIN_X = pd.DataFrame(np.random.random((len(SAMPLES), len(VARIANTS))), index=SAMPLES, columns=VARIANTS)


def _make_ancestry_model() -> AncestryModel:
    # one population per sample so that we can include all populations in train_y.
    pca_loadings_df = pd.DataFrame(
        np.random.random((len(VARIANTS), len(PC_COLUMNS))), index=VARIANTS, columns=PC_COLUMNS
    )
    train_Xpc = TRAIN_X @ pca_loadings_df
    train_y = POPS
    rfc = RandomForestClassifier(n_estimators=1, max_depth=1).fit(train_Xpc, train_y)
    return AncestryModel(pca_loadings_df, rfc)


handler_fn = handler_fn_factory(_make_ancestry_model())


def test_handler_fn_happy_path():
    progress_message = ProgressMessage(submissionID="my_submission_id")
    publisher = ProgressPublisher(
        host="127.0.0.1", port=1234, queue="my_queue", message=progress_message
    )
    ancestry_submission = AncestrySubmission("foo.vcf")
    ancestry_job_data = AncestryJobData(
        submissionID="my_submission_id2", ancestry_submission=ancestry_submission
    )
    samples = [f"sample{i}" for i in range(len(POPS))]
    variants = ["variant1", "variant2", "variant3"]
    train_X = pd.DataFrame(
        np.random.random((len(samples), len(variants))), index=samples, columns=variants
    )
    with patch("bystro.ancestry.listener._load_vcf", return_value=train_X) as _mock:
        handler_fn = handler_fn_factory(_make_ancestry_model())
        ancestry_response = handler_fn(publisher, ancestry_job_data)
    assert ancestry_submission.vcf_path == ancestry_response.vcf_path


def test_submit_msg_fn_happy_path():
    ancestry_submission = AncestrySubmission("foo.vcf")
    ancestry_job_data = AncestryJobData(
        submissionID="my_submission_id", ancestry_submission=ancestry_submission
    )
    submitted_job_message = submit_msg_fn(ancestry_job_data)
    assert submitted_job_message.submissionID == ancestry_job_data.submissionID


def test_completed_msg_fn_happy_path():
    progress_message = ProgressMessage(submissionID="my_submission_id")
    publisher = ProgressPublisher(
        host="127.0.0.1", port=1234, queue="my_queue", message=progress_message
    )

    ancestry_submission = AncestrySubmission("foo.vcf")
    ancestry_job_data = AncestryJobData(
        submissionID="my_submission_id", ancestry_submission=ancestry_submission
    )

    with patch("bystro.ancestry.listener._load_vcf", return_value=TRAIN_X) as _mock:
        ancestry_response = handler_fn(publisher, ancestry_job_data)
    ancestry_job_complete_message = completed_msg_fn(ancestry_job_data, ancestry_response)

    assert ancestry_job_complete_message.submissionID == ancestry_job_data.submissionID
    assert ancestry_job_complete_message.results == ancestry_response


def test_completed_msg_fn_rejects_nonmatching_vcf_paths():
    progress_message = ProgressMessage(submissionID="my_submission_id")
    publisher = ProgressPublisher(
        host="127.0.0.1", port=1234, queue="my_queue", message=progress_message
    )

    ancestry_submission = AncestrySubmission("foo.vcf")
    ancestry_job_data = AncestryJobData(
        submissionID="my_submission_id", ancestry_submission=ancestry_submission
    )

    with patch("bystro.ancestry.listener._load_vcf", return_value=TRAIN_X) as _mock:
        _correct_but_unused_ancestry_response = handler_fn(publisher, ancestry_job_data)

    progress_message = ProgressMessage(submissionID="my_submission_id")
    publisher = ProgressPublisher(
        host="127.0.0.1", port=1234, queue="my_queue", message=progress_message
    )

    # now instantiate another ancestry response with the wrong vcf...
    wrong_ancestry_submission = AncestrySubmission("bar.vcf")
    wrong_ancestry_job_data = AncestryJobData(
        submissionID="my_submission_id", ancestry_submission=wrong_ancestry_submission
    )

    with patch("bystro.ancestry.listener._load_vcf", return_value=TRAIN_X) as _:
        wrong_ancestry_response = handler_fn(publisher, wrong_ancestry_job_data)
    # end instantiating another ancestry response with the wrong vcf...

    with pytest.raises(
        ValueError, match="Ancestry submission filename .* doesn't match response filename"
    ):
        _ancestry_job_complete_message = completed_msg_fn(ancestry_job_data, wrong_ancestry_response)


def test_Ancestry_Model():
    ancestry_model = _make_ancestry_model()
    samples = [f"sample{i}" for i in range(len(POPS))]  # one pop per sample
    variants = ["variant1", "variant2", "variant3"]

    train_X = pd.DataFrame(
        np.random.random((len(samples), len(variants))), index=samples, columns=variants
    )

    pop_probs = ancestry_model.predict_proba(train_X)
    assert (pop_probs.index == samples).all()
    assert (pop_probs.columns == POPS).all()


def test__infer_ancestry():
    samples = [f"sample{i}" for i in range(len(POPS))]  # one pop per sample
    variants = ["variant1", "variant2", "variant3"]
    pc_columns = ["pc1", "pc2", "pc3", "pc4"]
    pca_loadings_df = pd.DataFrame(
        np.random.random((len(variants), len(pc_columns))), index=variants, columns=pc_columns
    )
    train_X = pd.DataFrame(
        np.random.random((len(samples), len(variants))), index=samples, columns=variants
    )
    train_Xpc = train_X @ pca_loadings_df
    train_y = POPS
    rfc = RandomForestClassifier(n_estimators=1, max_depth=1).fit(train_Xpc, train_y)
    ancestry_model = AncestryModel(pca_loadings_df, rfc)
    vcf_path = "my_vcf.vcf"
    ancestry_response = _infer_ancestry(ancestry_model, train_X, vcf_path)
    assert len(samples) == len(ancestry_response.results)
    assert vcf_path == ancestry_response.vcf_path
