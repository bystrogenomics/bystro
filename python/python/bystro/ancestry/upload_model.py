"""Upload PCA and RFC models to ancestry s3 bucket."""

import logging
import os
import sys
from pathlib import Path

import boto3

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

ANCESTRY_BUCKET = "bystro-ancestry"
ANCESTRY_MODEL_PRODUCTS_DIR = Path("ancestry_model_products")
PCA_FILE = "pca.csv"
RFC_FILE = "rfc.skop"

try:
    AWS_DEFAULT_PROFILE = os.environ["AWS_DEFAULT_PROFILE"]
    logger.info("found AWS_DEFAULT_PROFILE: %s", AWS_DEFAULT_PROFILE)
except KeyError as key_err:
    err_msg = (
        "AWS_DEFAULT_PROFILE not found in environment variables, "
        "check to see that it's defined and explicitly exported."
        "\n\n"
        "If you are running this script in an ipython process, os.environ may not see "
        "environment variables defined in another shell.  Try running from the command line instead."
    )
    raise RuntimeError(err_msg) from key_err


def upload_to_ancestry_bucket(filename: str) -> None:
    """Upload filename from ancestry model products dir to ancestry bucket."""
    logger.info("uploading %s...", filename)
    try:
        s3_client.upload_file(ANCESTRY_MODEL_PRODUCTS_DIR / filename, ANCESTRY_BUCKET, filename)
        logger.info("%s uploaded successfully", PCA_FILE)
    except Exception:
        logger.exception("Couldn't upload %s", filename)


if __name__ == "__main__":
    session = boto3.Session(profile_name=AWS_DEFAULT_PROFILE)
    s3_client = session.client("s3")
    logger.info("instantiated s3 client")

    upload_to_ancestry_bucket(PCA_FILE)
    upload_to_ancestry_bucket(RFC_FILE)
