from copy import deepcopy
from warnings import warn

import numpy as np
import pandas as pd

from bystro.proteomics.canopy.data.lift import LiftData


def calcELOD(x: pd.Series):
    """Calculated the estimated limit of detection based on x a pd.Series of SomaScan measurements of buffer."""
    med = np.median(x)
    absDiff = np.abs(x - med)
    medDiff = np.median(absDiff)
    eLOD = med + 3 * (1.4826 * medDiff)
    return eLOD


class AdatMathHelpers:
    """
    A collection of methods to help with performing common and standard computations on the adat.
    """

    plex_to_version = {'11K': 'v5.0', '7K': 'v4.1', '5K': 'v4.0'}

    def e_lod_by_reagent(self):
        df = self.pick_on_meta(axis=0, name='SampleType', values=['Buffer'])
        df.columns = df.columns.get_level_values('SeqId')
        e_lod_df = df.apply(calcELOD)
        return e_lod_df

    @staticmethod
    def _compute_intra_cv(adat, groupby):
        sums = {}
        for group_name, group_df in adat.groupby(level=groupby):
            group_means = group_df.mean(axis=0)
            square_diff_df = np.power(group_df - group_means, 2)
            sums[group_name] = square_diff_df.sum(axis=0)
        sums_df = pd.DataFrame(sums)
        df = pd.DataFrame(
            100 * np.sqrt(sums_df.sum(axis=1) / (adat.shape[0] - 1)) / adat.mean(axis=0)
        )
        return df.T

    @staticmethod
    def _compute_inter_cv(adat, groupby):
        sums = {}
        total_means = adat.mean(axis=0)
        for group_name, group_df in adat.groupby(level=groupby):
            group_means = group_df.mean(axis=0)
            square_diff_df = np.power(group_means - total_means, 2)
            sums[group_name] = square_diff_df * group_df.shape[0]
        sums_df = pd.DataFrame(sums)
        df = pd.DataFrame(
            100 * np.sqrt(sums_df.sum(axis=1) / (adat.shape[0] - 1)) / total_means
        )
        return df.T

    def cv_decomp(self, groupby=None, sample_type=None, sample_id=None):
        groupby = groupby or "PlateId"
        if sample_id:
            adat = self.xs(sample_id, level="SampleId")
        elif sample_type:
            adat = self.xs(sample_type, level="SampleType")
        else:
            adat = self.copy()

        intra_cv_df = self._compute_intra_cv(adat, groupby)
        inter_cv_df = self._compute_inter_cv(adat, groupby)
        total_cv_df = np.sqrt(np.power(intra_cv_df, 2) + np.power(inter_cv_df, 2))

        cv_decomps = {"Total": total_cv_df, "Intra": intra_cv_df, "Inter": inter_cv_df}
        cv_decomp_df = pd.concat(cv_decomps)
        cv_decomp_df.index = cv_decomp_df.index.droplevel(level=1)

        return cv_decomp_df.T

    def lift(self, lift_to_version: str = None):
        """Utility to perform lifting from one assay version to another on an ADAT data.
        Returns a new canopy.Adat object in the lifted assay RFU space.

        Reagents that are not in the prior version are lifted with the scalar 1.0, their value is unchanged.
        End users may need to drop these values to combine data with that of a study originating with the previous assay version.

        Parameters
        ----------
        lift_to_version: str. The SomaScan version you would like to lift the current ADAT to.

        Returns
        -------
        lifted_adat : Adat
            Canopy Adat object with scaled RFU.

        Examples
        --------
        >>> # the adat stores the current assay version. This value is used by the tool to select the correct reference but you are not required to ender it.
        >>> adat.header_metadata['!AssayVersion']
        'v5.0'
        >>> # the adat stores the matrix. This value is used by the tool to select the correct reference but you are not required to ender it.
        >>> adat.header_metadata['StudyMatrix']
        'EDTA Plasma'
        >>> lifted_adat = adat.lift('v4.1') # lifting to the previous assay version.
        """
        adat = self.copy()
        # Perform checks to see if this bridging is appropriate for this adat
        process_steps = adat.header_metadata['!ProcessSteps']
        if not 'anmlsmp' in process_steps.lower():
            raise ValueError(
                f'ANML normalized SOMAscan data is required for lifting. Provided norm steps: "{process_steps}"'
            )

        # Get matrix from adat header metadata
        try:
            matrix = adat.header_metadata['!StudyMatrix']
        except KeyError:
            matrix = adat.header_metadata['StudyMatrix']
        if (
            matrix == 'EDTA Plasma'
        ):  # Takes care of the EDTA Plasma --> Plasma conversion so we can look up the column in the annotations df
            matrix = 'Plasma'
        # Get assay version from adat header metadata. Prefer SignalSpace (created by lifting apps) if it exists
        if 'SignalSpace' in adat.header_metadata.keys():
            signal_space = adat.header_metadata['SignalSpace']
        else:
            signal_space = adat.header_metadata['!AssayVersion']
        if (
            signal_space.lower() == 'v4'
        ):  # Takes care of the v4 and V4 --> v4.0 conversion so we can look up the column in the annotations df
            signal_space = 'v4.0'

        # go from assay version (vx.y) to plex (zK) namespace
        if signal_space in self.plex_to_version.keys():
            signal_space = self.plex_to_version[signal_space]
        if lift_to_version in self.plex_to_version.keys():
            lift_to_version = self.plex_to_version[lift_to_version]

        # instantiate lift object to get scale factors.
        lift = LiftData(signal_space, lift_to_version, matrix)

        # Check if seq ids will broadcast between adat & annotations (symmetric difference)
        adat_cols = set(adat.columns.get_level_values('SeqId'))
        scalar_index = set(lift.scale_factors.index)
        sym_diff = scalar_index ^ adat_cols
        if sym_diff:
            if all([x in adat_cols for x in scalar_index]):
                warn(
                    'ADAT contains reagents that do not have lifting scalars. These sequences will be scaled by 1.0.'
                )
                # All the scalars apply to a column in the ADAT. There are others columns in the adat that won't be scaled set to 1.0
                extra = pd.Series(np.ones(len(sym_diff)), index=sym_diff)
                lift.scale_factors = pd.concat([lift.scale_factors, extra])
                lift.scale_factors = lift.scale_factors.loc[
                    adat.columns.get_level_values('SeqId')
                ]
            elif all([x in scalar_index for x in adat_cols]):
                # the adat is a subset of the scalars there are extra scalars that can't be used. Shrink the scalars.
                warn(
                    "The ADAT contains a subset of the sequence in the references.  This file may be missing measurements in the lifted RFU space."
                )
                lift.scale_factors = lift.scale_factors.loc[
                    adat.columns.get_level_values('SeqId')
                ]
            else:
                # there are differences on both sides. This is potentially a mismatch. Fail and inform the user.
                raise ValueError(
                    'Unable to perform lifting due to analyte mismatch between adat & lift reference unable to lift.'
                )

        # Scale adat
        scaled_adat = adat.multiply(
            lift.scale_factors, axis='columns', level='SeqId'
        ).round(1)
        scaled_adat.header_metadata = deepcopy(adat.header_metadata)
        scaled_adat.header_metadata[
            '!ProcessSteps'
        ] += f', Lifting Bridge ({signal_space} -> {lift_to_version})'
        scaled_adat.header_metadata['SignalSpace'] = lift_to_version
        return scaled_adat
