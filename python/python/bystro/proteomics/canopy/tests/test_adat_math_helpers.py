import os
from numpy import isclose

import bystro.proteomics.canopy as canopy
from bystro.proteomics.canopy.data.lift import LiftData


class TestAdatLift:
    filename = os.path.join(os.path.dirname(__file__), "../data", "control_data.adat")

    def __init__(self):
        self.ld = LiftData("v4.0", "v5.0", "plasma")
        self.adat = canopy.read_adat(self.filename)
        self.lift_adat = self.adat.lift("v5.0")

    def test_scaling_math(self):
        back_scalers = self.lift_adat.divide(self.adat).mean(axis=0)
        # theres rounding error but we're lifting columns. The Ratio of lifted values is the scale factors again.
        assert all(isclose(back_scalers.values, self.ld.scale_factors.values, atol=0.05))

    def test_processing_steps(self):
        assert self.lift_adat.header_metadata["!ProcessSteps"].endswith("Lifting Bridge (v4.0 -> v5.0)")

    def test_signal_space(self):
        assert self.lift_adat.header_metadata["SignalSpace"] == "v5.0"
