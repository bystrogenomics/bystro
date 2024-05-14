"""Load and prep fragpipe tandem mass Tag datasets."""

import canopy  # type: ignore

from msgspec import Struct


class SomascanDataset(Struct):
    """Represent a Fragpipe Tandem Mass Tag dataset."""

    adat: canopy.Adat
    annotations: canopy.Annotations | None = None

    @staticmethod
    def from_paths(adat_path: str, annotations_path: str | None = None) -> "SomascanDataset":
        """
        Load a SomascanDataset from an adat file and an optional annotations file.
        """
        adat = canopy.read_adat(adat_path)
        annotations = None

        if annotations_path:
            annotations = canopy.read_annotations(annotations_path)

        return SomascanDataset(adat, annotations)

    def __post_init__(self) -> None:
        if not isinstance(self.adat, canopy.Adat):
            raise ValueError(
                f"`adat` argument to SomascanDataset must be a canopy.Adat object, not {type(self.adat)}"
            )

        if self.annotations and not isinstance(self.annotations, canopy.Annotations):
            raise ValueError(
                (
                    "`annotations` argument to SomascanDataset must be a canopy.Annotations object"
                    f", not {type(self.annotations)}"
                )
            )
