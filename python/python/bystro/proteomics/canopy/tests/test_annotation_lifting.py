from io import StringIO
from unittest import TestCase

import pandas as pd

from bystro.proteomics.canopy import Adat, Annotations
from bystro.proteomics.canopy.errors import AnnotationsLiftingError


def build_annotation_example():
    s = StringIO(
        'SeqId,SomaId,Plasma Scalar v4.0 5K to v4.1 7K\n54321-21,SL054321,0.8\n12345-12,SL012345,1.1\n'
    )
    df = pd.read_csv(s, index_col=False)
    return Annotations(data=df.values, index=df.index, columns=df.columns)


def build_good_example_adat():
    rfu_data = [[1, 2], [4, 5]]
    col_metadata = {
        'SeqId': ['12345-12', '54321-21'],
        'SeqIdVersion': ['1', '2'],
        'ColCheck': ['PASS', 'FLAG'],
    }
    row_metadata = {'PlateId': ['A12', 'A12'], 'Barcode': ['SL1234', 'SL1235']}
    header_metadata = {
        'AdatId': '1a2b3c',
        '!ProcessSteps': 'Raw RFU, Hyb Normalization, medNormInt (SampleId), plateScale, Calibration, anmlQC, qcCheck, anmlSMP',
        'StudyMatrix': 'EDTA Plasma',
        '!AssayVersion': 'v4.0',
    }
    return Adat.from_features(rfu_data, row_metadata, col_metadata, header_metadata)


def build_example_adat_with_extra_somamers():
    rfu_data = [[1, 2, 3], [4, 5, 6]]
    col_metadata = {
        'SeqId': ['12345-12', '54321-21', '23456-78'],
        'SeqIdVersion': ['1', '2', '3'],
        'ColCheck': ['PASS', 'FLAG', 'FLAG'],
    }
    row_metadata = {'PlateId': ['A12', 'A12'], 'Barcode': ['SL1234', 'SL1235']}
    header_metadata = {
        'AdatId': '1a2b3c',
        '!ProcessSteps': 'Raw RFU, Hyb Normalization, medNormInt (SampleId), plateScale, Calibration, anmlQC, qcCheck, anmlSMP',
        'StudyMatrix': 'EDTA Plasma',
        '!AssayVersion': 'v4.0',
    }
    return Adat.from_features(rfu_data, row_metadata, col_metadata, header_metadata)


class AdatLiftingTestCase(TestCase):
    def setUp(self):
        self.an = build_annotation_example()
        self.adat = build_good_example_adat()

    def test_lifting_correct(self):
        lifted_adat = self.an.lift_adat(self.adat)
        correct_lifted_values = [[1.1, 1.6], [4.4, 4]]
        for correct_row, lifted_row in zip(correct_lifted_values, lifted_adat.values):
            for correct_value, lifted_value in zip(correct_row, lifted_row):
                self.assertAlmostEqual(correct_value, lifted_value)

        self.assertEqual(lifted_adat.header_metadata['SignalSpace'], 'v4.1')
        self.assertTrue(
            lifted_adat.header_metadata['!ProcessSteps'].endswith(
                ', Lifting Bridge (v4.0 -> v4.1)'
            )
        )

    def test_orig_adat_unmodified(self):
        self.an.lift_adat(self.adat)

        orig_values = [[1, 2], [4, 5]]
        for orig_row, adat_row in zip(orig_values, self.adat.values):
            for orig_value, adat_value in zip(orig_row, adat_row):
                self.assertAlmostEqual(orig_value, adat_value)

        self.assertNotIn('SignalSpace', self.adat.header_metadata)
        self.assertFalse(
            self.adat.header_metadata['!ProcessSteps'].endswith(
                ', Lifting Bridge (v4.0 to v4.1)'
            )
        )

    def test_lifting_matrix_error(self):
        self.adat.header_metadata['StudyMatrix'] = 'NotSupportedMatrix'
        with self.assertRaises(AnnotationsLiftingError) as cm:
            self.an.lift_adat(self.adat)
        e_msg = str(cm.exception)
        self.assertEqual(
            e_msg,
            'Unsupported matrix: "NotSupportedMatrix". Supported matrices: Plasma.',
        )

    def test_lifting_assay_version_error(self):
        self.adat.header_metadata['!AssayVersion'] = 'NotSupportedVersion'
        with self.assertRaises(AnnotationsLiftingError) as cm:
            self.an.lift_adat(self.adat)
        e_msg = str(cm.exception)
        self.assertEqual(
            e_msg,
            'Unsupported lifting from: "NotSupportedVersion". Supported lifting: from "v4.0" to "v4.1".',
        )

    def test_lifting_signal_space_error(self):
        self.adat.header_metadata['SignalSpace'] = 'NotSupportedVersion'
        with self.assertRaises(AnnotationsLiftingError) as cm:
            self.an.lift_adat(self.adat)
        e_msg = str(cm.exception)
        self.assertEqual(
            e_msg,
            'Unsupported lifting from: "NotSupportedVersion". Supported lifting: from "v4.0" to "v4.1".',
        )

    def test_lifting_to_space_error(self):
        with self.assertRaises(AnnotationsLiftingError) as cm:
            self.an.lift_adat(self.adat, lift_to_version='NotSupportedVersion')
        e_msg = str(cm.exception)
        self.assertEqual(
            e_msg,
            'Unsupported lifting from "v4.0" to "NotSupportedVersion". Supported lifting: from "v4.0" to "v4.1".',
        )

    def test_analyte_mismatch(self):
        self.adat = build_example_adat_with_extra_somamers()
        with self.assertRaises(AnnotationsLiftingError) as cm:
            self.an.lift_adat(self.adat)
        e_msg = str(cm.exception)
        self.assertEqual(
            e_msg,
            'Unable to perform lifting due to analyte mismatch between adat & annotations. Has either file been modified?',
        )
