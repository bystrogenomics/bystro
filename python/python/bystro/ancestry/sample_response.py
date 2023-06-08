from bystro.ancestry.ancestry_types import AncestryResponse

json_payload = """{"vcf_path": "myfile.vcf", "results": [{"sample_id":
"foo", "populations": {"ACB": {"lower_bound": 0.0, "upper_bound":
1.0}, "ASW": {"lower_bound": 0.0, "upper_bound": 1.0}, "BEB":
{"lower_bound": 0.0, "upper_bound": 1.0}, "CDX": {"lower_bound": 0.0,
"upper_bound": 1.0}, "CEU": {"lower_bound": 0.0, "upper_bound": 1.0},
"CHB": {"lower_bound": 0.0, "upper_bound": 1.0}, "CHS":
{"lower_bound": 0.0, "upper_bound": 1.0}, "CLM": {"lower_bound": 0.0,
"upper_bound": 1.0}, "ESN": {"lower_bound": 0.0, "upper_bound": 1.0},
"FIN": {"lower_bound": 0.0, "upper_bound": 1.0}, "GBR":
{"lower_bound": 0.0, "upper_bound": 1.0}, "GIH": {"lower_bound": 0.0,
"upper_bound": 1.0}, "GWD": {"lower_bound": 0.0, "upper_bound": 1.0},
"IBS": {"lower_bound": 0.0, "upper_bound": 1.0}, "ITU":
{"lower_bound": 0.0, "upper_bound": 1.0}, "JPT": {"lower_bound": 0.0,
"upper_bound": 1.0}, "KHV": {"lower_bound": 0.0, "upper_bound": 1.0},
"LWK": {"lower_bound": 0.0, "upper_bound": 1.0}, "MAG":
{"lower_bound": 0.0, "upper_bound": 1.0}, "MSL": {"lower_bound": 0.0,
"upper_bound": 1.0}, "MXL": {"lower_bound": 0.0, "upper_bound": 1.0},
"PEL": {"lower_bound": 0.0, "upper_bound": 1.0}, "PJL":
{"lower_bound": 0.0, "upper_bound": 1.0}, "PUR": {"lower_bound": 0.0,
"upper_bound": 1.0}, "STU": {"lower_bound": 0.0, "upper_bound": 1.0},
"TSI": {"lower_bound": 0.0, "upper_bound": 1.0}, "YRI":
{"lower_bound": 0.0, "upper_bound": 1.0}}, "superpops": {"AFR":
{"lower_bound": 0.0, "upper_bound": 1.0}, "AMR": {"lower_bound": 0.0,
"upper_bound": 1.0}, "EAS": {"lower_bound": 0.0, "upper_bound": 1.0},
"EUR": {"lower_bound": 0.0, "upper_bound": 1.0}, "SAS":
{"lower_bound": 0.0, "upper_bound": 1.0}}, "missingness": 0.5},
{"sample_id": "bar", "populations": {"ACB": {"lower_bound": 0.0,
"upper_bound": 1.0}, "ASW": {"lower_bound": 0.0, "upper_bound": 1.0},
"BEB": {"lower_bound": 0.0, "upper_bound": 1.0}, "CDX":
{"lower_bound": 0.0, "upper_bound": 1.0}, "CEU": {"lower_bound": 0.0,
"upper_bound": 1.0}, "CHB": {"lower_bound": 0.0, "upper_bound": 1.0},
"CHS": {"lower_bound": 0.0, "upper_bound": 1.0}, "CLM":
{"lower_bound": 0.0, "upper_bound": 1.0}, "ESN": {"lower_bound": 0.0,
"upper_bound": 1.0}, "FIN": {"lower_bound": 0.0, "upper_bound": 1.0},
"GBR": {"lower_bound": 0.0, "upper_bound": 1.0}, "GIH":
{"lower_bound": 0.0, "upper_bound": 1.0}, "GWD": {"lower_bound": 0.0,
"upper_bound": 1.0}, "IBS": {"lower_bound": 0.0, "upper_bound": 1.0},
"ITU": {"lower_bound": 0.0, "upper_bound": 1.0}, "JPT":
{"lower_bound": 0.0, "upper_bound": 1.0}, "KHV": {"lower_bound": 0.0,
"upper_bound": 1.0}, "LWK": {"lower_bound": 0.0, "upper_bound": 1.0},
"MAG": {"lower_bound": 0.0, "upper_bound": 1.0}, "MSL":
{"lower_bound": 0.0, "upper_bound": 1.0}, "MXL": {"lower_bound": 0.0,
"upper_bound": 1.0}, "PEL": {"lower_bound": 0.0, "upper_bound": 1.0},
"PJL": {"lower_bound": 0.0, "upper_bound": 1.0}, "PUR":
{"lower_bound": 0.0, "upper_bound": 1.0}, "STU": {"lower_bound": 0.0,
"upper_bound": 1.0}, "TSI": {"lower_bound": 0.0, "upper_bound": 1.0},
"YRI": {"lower_bound": 0.0, "upper_bound": 1.0}}, "superpops": {"AFR":
{"lower_bound": 0.0, "upper_bound": 1.0}, "AMR": {"lower_bound": 0.0,
"upper_bound": 1.0}, "EAS": {"lower_bound": 0.0, "upper_bound": 1.0},
"EUR": {"lower_bound": 0.0, "upper_bound": 1.0}, "SAS":
{"lower_bound": 0.0, "upper_bound": 1.0}}, "missingness": 0.5},
{"sample_id": "baz", "populations": {"ACB": {"lower_bound": 0.0,
"upper_bound": 1.0}, "ASW": {"lower_bound": 0.0, "upper_bound": 1.0},
"BEB": {"lower_bound": 0.0, "upper_bound": 1.0}, "CDX":
{"lower_bound": 0.0, "upper_bound": 1.0}, "CEU": {"lower_bound": 0.0,
"upper_bound": 1.0}, "CHB": {"lower_bound": 0.0, "upper_bound": 1.0},
"CHS": {"lower_bound": 0.0, "upper_bound": 1.0}, "CLM":
{"lower_bound": 0.0, "upper_bound": 1.0}, "ESN": {"lower_bound": 0.0,
"upper_bound": 1.0}, "FIN": {"lower_bound": 0.0, "upper_bound": 1.0},
"GBR": {"lower_bound": 0.0, "upper_bound": 1.0}, "GIH":
{"lower_bound": 0.0, "upper_bound": 1.0}, "GWD": {"lower_bound": 0.0,
"upper_bound": 1.0}, "IBS": {"lower_bound": 0.0, "upper_bound": 1.0},
"ITU": {"lower_bound": 0.0, "upper_bound": 1.0}, "JPT":
{"lower_bound": 0.0, "upper_bound": 1.0}, "KHV": {"lower_bound": 0.0,
"upper_bound": 1.0}, "LWK": {"lower_bound": 0.0, "upper_bound": 1.0},
"MAG": {"lower_bound": 0.0, "upper_bound": 1.0}, "MSL":
{"lower_bound": 0.0, "upper_bound": 1.0}, "MXL": {"lower_bound": 0.0,
"upper_bound": 1.0}, "PEL": {"lower_bound": 0.0, "upper_bound": 1.0},
"PJL": {"lower_bound": 0.0, "upper_bound": 1.0}, "PUR":
{"lower_bound": 0.0, "upper_bound": 1.0}, "STU": {"lower_bound": 0.0,
"upper_bound": 1.0}, "TSI": {"lower_bound": 0.0, "upper_bound": 1.0},
"YRI": {"lower_bound": 0.0, "upper_bound": 1.0}}, "superpops": {"AFR":
{"lower_bound": 0.0, "upper_bound": 1.0}, "AMR": {"lower_bound": 0.0,
"upper_bound": 1.0}, "EAS": {"lower_bound": 0.0, "upper_bound": 1.0},
"EUR": {"lower_bound": 0.0, "upper_bound": 1.0}, "SAS":
{"lower_bound": 0.0, "upper_bound": 1.0}}, "missingness": 0.5}]}"""

sample_ancestry_response = AncestryResponse.parse_raw(json_payload)
