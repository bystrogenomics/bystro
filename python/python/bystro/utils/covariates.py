from msgspec import Struct, structs


class Covariates(Struct, forbid_unknown_fields=False, rename="camel", kw_only=True):
    """
    Covariates that Bystro natively understands, and may use for
    downstream analysis.  This is a subset of the covariates that
    may be present in the user's data.
    """

    sex: str | None = None
    phenotype: str | None = None
    population: str | None = None
    superpopulation: str | None = None


class ExperimentMapping(Struct, rename="camel", kw_only=True):
    sample_id: str
    subject_id: str | None = None
    covariates: Covariates


class ExperimentMappings(Struct, rename="camel", kw_only=True):
    user_experiment_mappings: list[ExperimentMapping]

    @classmethod
    def from_path(cls, path: str) -> "ExperimentMappings":
        """
        Read the experiment mappings from a tab-separated file

        The first row is expected to be the header, with the following columns:
            1. sample_id
            2. subject_id
            3..n. covariates 1 ... n
        """
        with open(path, "r") as f:
            lines = f.readlines()

        header = [h.lower() for h in lines[0].strip().split("\t")]

        # find the sample_id, subject_id columns
        try:
            sample_id_index = header.index("sample_id")
        except ValueError:
            raise ValueError("sample_id not found in the header of the covariate file")

        user_experiment_mappings: list[ExperimentMapping] = []
        line_idx = 0
        for line in lines[1:]:
            line_idx += 1
            fields = line.strip().split("\t")

            sample_id = fields[sample_id_index]

            if sample_id == "" or sample_id is None:
                raise ValueError(
                    "sample_id cannot be empty, found in the covariate file on row %d" % line_idx
                )

            covariates = {}
            for i, field in enumerate(fields):
                if i == sample_id_index:
                    continue

                for cov_field_info in structs.fields(Covariates):
                    if cov_field_info.name == header[i]:
                        covariates[header[i]] = field

            user_experiment_mappings.append(
                ExperimentMapping(sample_id=sample_id, covariates=Covariates(**covariates))
            )

        return cls(user_experiment_mappings=user_experiment_mappings)
