import argparse
import io

import msgspec

from bystro.api.ancestry import calculate_ancestry_scores, ancestry_json_to_format


def add_ancestry_subparser(subparsers):
    parser = subparsers.add_parser(
        "ancestry",
        help="Ancestry analysis fun",
    )

    ancestry_subparser = parser.add_subparsers(
        title="commands", dest="command", help="Commands for ancestry analysis"
    )

    def ancestry_default_help(_args):
        buffer = io.StringIO()
        parser.print_help(file=buffer)
        help_message = buffer.getvalue()
        custom_message = "Process VCF files with bystro-vcf and perform ancestry analysis.\n\n"
        insertion_point = help_message.find("options:")
        if insertion_point != -1:
            help_message = (
                help_message[:insertion_point] + custom_message + help_message[insertion_point:]
            )
        else:
            help_message = custom_message + help_message
        print(help_message)
        parser.exit()

    parser.set_defaults(func=ancestry_default_help)

    ancestry_score_parser = ancestry_subparser.add_parser(
        "score",
        help=(
            "Using a vcf file, calculate the ancestry scores of the samples in the vcf file "
            "using the best-fitting ancestry model for the given assembly version."
        ),
    )

    ancestry_score_parser.add_argument(
        "--in", dest="input_vcf", type=str, required=True, help="The input VCF file"
    )
    ancestry_score_parser.add_argument(
        "--assembly",
        type=str,
        required=True,
        choices=["hg19", "hg38"],
        help="The assembly version (hg19 or hg38)",
    )
    ancestry_score_parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "The output directory for the ancestry results. "
            "If blank, the results are written to stdout. (optional)"
        ),
    )
    ancestry_score_parser.add_argument(
        "--dosage",
        default=False,
        action="store_true",
        help="Whether or not to write the dosage matrix output to the output directory (optional)",
    )

    ancestry_score_parser.set_defaults(func=_calculate_and_write_ancestry_scores)

    # Add new subparser for JSON to TSV/CSV conversion
    convert_parser = ancestry_subparser.add_parser(
        "convert", help="Convert JSON ancestry results to TSV or CSV format"
    )

    convert_parser.add_argument("--input-json", type=str, required=True, help="The input JSON file")
    convert_parser.add_argument(
        "--output", type=str, required=True, help="The output file (TSV or CSV)"
    )
    convert_parser.add_argument(
        "--format",
        type=str,
        choices=["tsv", "csv"],
        default="tsv",
        help="The output format (default is tsv)",
    )

    convert_parser.set_defaults(func=_convert_json_to_tsv_or_excel)


def _calculate_and_write_ancestry_scores(args: argparse.Namespace):
    """Calculate ancestry scores from a VCF file and write the results to a file.

    Args:
        args (argparse.Namespace): The parsed arguments from the command line.
         Arguments expected:
            - assembly: str
                The assembly version (hg19 or hg38)
            - input_vcf: str
                The input VCF file
            - out: str | None
                The output directory for the ancestry results.
                If not provided, the results are written to stdout.
            - dosage: bool
                Whether or not to write the dosage matrix output to the output directory.

    Returns:
        None
    """
    res = calculate_ancestry_scores(
        vcf=args.input_vcf, assembly=args.assembly, dosage=args.dosage, out_dir=args.out_dir
    )

    if args.out_dir is None:
        json_data = msgspec.json.encode(res)

        print(str(json_data, "utf-8"))

    else:
        print(f"Ancestry results written to {args.out_dir}")


def _convert_json_to_tsv_or_excel(args: argparse.Namespace):
    """Convert JSON ancestry results to TSV or CSV format

    Args:
        args (argparse.Namespace): The parsed arguments from the command line.
         Arguments expected:
            - input_json: str
                The input JSON file
            - output: str
                The path to the output TSV or CSV file
            - format: str
                The output format (TSV or CSV)

    Returns:
        None
    """
    ancestry_json_to_format(args.input_json, args.output, args.format)
    print(f"\n\nConverted {args.input_json} to {args.format.upper()} format at {args.output}\n")
