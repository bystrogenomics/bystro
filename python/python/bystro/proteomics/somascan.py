"""Load and prep fragpipe tandem mass Tag datasets."""

import canopy

from msgspec import Struct


class SomascanDataset(Struct):
    """Represent a Fragpipe Tandem Mass Tag dataset."""

    adat: canopy.Adat
    annotations: canopy.Annotations | None

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
