import argparse
import io

from bystro.api.proteomics import upload_proteomics_dataset


def upload_proteomics_cli(args: argparse.Namespace) -> dict:
    """
    Upload a fragpipe-TMT dataset through the /api/jobs/proteomics/ endpoint and
    update the annotation job.

    Parameters
    ----------
    protein_abundance_file : str
        Path to the protein abundance file.
    experiment_annotation_file : str | None
        Path to the experiment annotation file.
    annotation_job_id : str | None
        annotationId of the job associated with the annotation dataset.
    experiment_name : str | None
        Name of the experiment, required if the experiment annotation file contains multiple experiments
    print_result : bool, optional
        Whether to print the result of the upload operation, by default True.

    Returns
    -------
    dict
        A json response with annotationID and proteomicsID.
    """

    return upload_proteomics_dataset(
        protein_abundance_file=args.protein_abundance_file,
        experiment_annotation_file=args.experiment_annotation_file,
        annotation_job_id=args.annotation_job_id,
        experiment_name=args.experiment_name,
        print_result=True,
    )


def add_proteomics_subparser(subparsers) -> None:
    proteomics_parser = subparsers.add_parser("proteomics", help="Bystro proteomics dataset operations")
    proteomics_subparser = proteomics_parser.add_subparsers(
        title="commands", dest="command", help="Commands for proteomics datasets"
    )

    def proteomics_default_help(_args):
        buffer = io.StringIO()
        proteomics_parser.print_help(file=buffer)
        help_message = buffer.getvalue()
        custom_message = "Bystro proteomics dataset operations.\n\n"
        insertion_point = help_message.find("options:")
        if insertion_point != -1:
            help_message = (
                help_message[:insertion_point] + custom_message + help_message[insertion_point:]
            )
        else:
            help_message = custom_message + help_message
        print(help_message)
        proteomics_parser.exit()

    proteomics_parser.set_defaults(func=proteomics_default_help)

    upload_parser = proteomics_subparser.add_parser("upload", help="Upload a proteomics dataset")
    upload_parser.add_argument(
        "--protein_abundance_file", required=True, help="Path to the protein abundance file"
    )
    upload_parser.add_argument(
        "--experiment_annotation_file", help="Path to the experiment annotation file"
    )
    upload_parser.add_argument(
        "--annotation_job_id", help="ID of the annotation job to link with the proteomics dataset"
    )
    upload_parser.add_argument(
        "--experiment_name",
        help=("Name of the experiment, required if the experiment "
              "annotation file contains multiple experiments")
    )
    upload_parser.set_defaults(func=upload_proteomics_cli)
