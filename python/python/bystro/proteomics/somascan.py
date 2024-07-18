"""Load and prep fragpipe tandem mass Tag datasets."""

import somadata  # type: ignore

from msgspec import Struct

ADAT_GENE_NAME_COLUMN = "Target"
ADAT_PROTEIN_NAME_COLUMN = "UniProt"
ADAT_SAMPLE_ID_COLUMN = "SampleId"
ADAT_RFU_COLUMN = "RFU"

class SomascanDataset(Struct):
    """Represent a SomaScan Aptamer dataset."""

    adat: somadata.Adat
    annotations: somadata.Annotations | None = None

    @staticmethod
    def from_paths(adat_path: str, annotations_path: str | None = None) -> "SomascanDataset":
        """
        Load a SomascanDataset from an adat file and an optional annotations file.
        """
        adat = somadata.read_adat(adat_path)
        annotations = None

        if annotations_path:
            annotations = somadata.read_annotations(annotations_path)

        return SomascanDataset(adat, annotations)

    def __post_init__(self) -> None:
        if not isinstance(self.adat, somadata.Adat):
            raise ValueError(
                (
                    "`adat` argument to SomascanDataset must be a "
                    f"somadata.Adat object, not {type(self.adat)}"
                )
            )

        if self.annotations and not isinstance(self.annotations, somadata.Annotations):
            raise ValueError(
                (
                    "`annotations` argument to SomascanDataset must be a somadata.Annotations object"
                    f", not {type(self.annotations)}"
                )
            )

    def to_melted_frame(self):
        return self.adat.stack( # noqa: PD013
            level=list(range(self.adat.columns.nlevels)), future_stack=True
        ).reset_index(name=ADAT_RFU_COLUMN)
