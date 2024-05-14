import os

from unittest import TestCase

import pandas as pd
import pytest

import bystro.proteomics.canopy as canopy
from bystro.proteomics.canopy import Adat
from bystro.proteomics.canopy.errors import AdatMetaError

control_dat_path = os.path.join(os.path.dirname(__file__), "../data", "control_data.adat")


class ExcludeMetaTest(TestCase):
    def setUp(self):
        self.adat = canopy.read_adat(control_dat_path)

    def test_exclude_removes_row_metadata(self):
        self.assertIn("PlateId", self.adat.index.names)
        adat = self.adat.exclude_meta(axis=0, names=["PlateId"])
        self.assertNotIn("PlateId", adat.index.names)

    def test_exclude_removes_column_metadata(self):
        self.assertIn("Type", self.adat.columns.names)
        adat = self.adat.exclude_meta(axis=1, names=["Type"])
        self.assertNotIn("Type", adat.columns.names)

    def test_exclude_removes_metadata_rows(self):
        metadata = set(self.adat.index.names)
        metadata_to_exclude = set(["PlateId", "Barcode"])
        intersection = metadata.intersection(metadata_to_exclude)
        self.assertEqual(intersection, metadata_to_exclude)

        adat = self.adat.exclude_meta(axis=0, names=metadata_to_exclude)
        new_metadata = set(adat.index.names)
        new_intersection = new_metadata.intersection(metadata_to_exclude)
        self.assertEqual(len(new_intersection), 0)


class ExcludeOnMetaTest(TestCase):
    def setUp(self):
        self.adat = canopy.read_adat(control_dat_path)

    def test_exclude_on_runs(self):
        self.adat.exclude_on_meta(axis=0, name="PlatePosition", values=["H8"])
        self.adat.exclude_on_meta(axis=1, name="Type", values=["Spuriomer"])

    def test_exclude_on_removes_row(self):
        self.assertIn("H8", self.adat.index.get_level_values("PlatePosition"))
        adat = self.adat.exclude_on_meta(axis=0, name="PlatePosition", values=["H8"])
        self.assertNotIn("H8", adat.index.get_level_values("PlatePosition"))

    def test_exclude_on_removes_column(self):
        self.assertIn("10000-28", self.adat.columns.get_level_values("SeqId"))
        adat = self.adat.exclude_on_meta(axis=1, name="SeqId", values=["10000-28"])
        self.assertNotIn("10000-28", adat.columns.get_level_values("SeqId"))

    def test_exclude_on_removes_rows(self):
        self.assertIn("Calibrator", self.adat.index.get_level_values("SampleType"))
        adat = self.adat.exclude_on_meta(axis=0, name="SampleType", values=["Calibrator"])
        self.assertNotIn("Calibrator", adat.index.get_level_values("SampleType"))

    def test_exclude_on_removes_columns(self):
        self.assertIn("Protein", self.adat.columns.get_level_values("Type"))
        adat = self.adat.exclude_on_meta(axis=1, name="Type", values=["Protein"])
        self.assertNotIn("Protein", adat.columns.get_level_values("Type"))


class PickMetaTest(TestCase):
    def setUp(self):
        self.adat = canopy.read_adat(control_dat_path)

    def test_pick_meta_row(self):
        self.assertIn("PlateId", self.adat.index.names)
        self.assertIn("SampleId", self.adat.index.names)
        adat = self.adat.pick_meta(axis=0, names=["PlateId"])
        self.assertIn("PlateId", adat.index.names)
        self.assertEqual(len(adat.index.names), 1)
        self.assertNotIn("SampleId", adat.index.names)

    def test_pick_meta_rows(self):
        self.assertIn("PlateId", self.adat.index.names)
        self.assertIn("SampleId", self.adat.index.names)
        self.assertIn("Barcode", self.adat.index.names)
        adat = self.adat.pick_meta(axis=0, names=["PlateId", "Barcode"])
        self.assertIn("PlateId", adat.index.names)
        self.assertIn("Barcode", adat.index.names)
        self.assertEqual(len(adat.index.names), 2)
        self.assertNotIn("SampleId", adat.index.names)

    def test_pick_meta_column(self):
        self.assertIn("SeqId", self.adat.columns.names)
        self.assertIn("Type", self.adat.columns.names)
        adat = self.adat.pick_meta(axis=1, names=["SeqId"])
        self.assertIn("SeqId", adat.columns.names)
        self.assertEqual(len(adat.columns.names), 1)
        self.assertNotIn("Type", adat.columns.names)

    def test_pick_meta_columns(self):
        self.assertIn("SeqId", self.adat.columns.names)
        self.assertIn("Type", self.adat.columns.names)
        self.assertIn("Dilution", self.adat.columns.names)
        adat = self.adat.pick_meta(axis=1, names=["SeqId", "Dilution"])
        self.assertIn("SeqId", adat.columns.names)
        self.assertIn("Dilution", adat.columns.names)
        self.assertEqual(len(adat.columns.names), 2)
        self.assertNotIn("Type", adat.columns.names)


class PickOnMetaTest(TestCase):
    def setUp(self):
        self.adat = canopy.read_adat(control_dat_path)

    def test_pick_on_row(self):
        self.assertIn("H8", self.adat.index.get_level_values("PlatePosition"))
        self.assertIn("G7", self.adat.index.get_level_values("PlatePosition"))
        adat = self.adat.pick_on_meta(axis=0, name="PlatePosition", values=["H8"])
        self.assertIn("H8", adat.index.get_level_values("PlatePosition"))
        self.assertNotIn("G7", adat.index.get_level_values("PlatePosition"))
        self.assertEqual(len(adat.index.get_level_values("PlatePosition")), 1)

    def test_pick_on_row_raises(self):
        self.assertNotIn("H50", self.adat.index.get_level_values("PlatePosition"))
        with self.assertRaises(KeyError):
            self.adat.pick_on_meta(axis=0, name="PlatePosition", values=["H50"])

    def test_pick_on_column(self):
        self.assertIn("10000-28", self.adat.columns.get_level_values("SeqId"))
        self.assertIn("10001-7", self.adat.columns.get_level_values("SeqId"))
        adat = self.adat.pick_on_meta(axis=1, name="SeqId", values=["10000-28"])
        self.assertIn("10000-28", adat.columns.get_level_values("SeqId"))
        self.assertNotIn("10001-7", adat.columns.get_level_values("SeqId"))
        self.assertEqual(len(adat.columns.get_level_values("SeqId")), 1)

    def test_pick_on_rows(self):
        self.assertIn("Calibrator", self.adat.index.get_level_values("SampleType"))
        self.assertIn("Buffer", self.adat.index.get_level_values("SampleType"))
        self.assertIn("QC", self.adat.index.get_level_values("SampleType"))
        adat = self.adat.pick_on_meta(axis=0, name="SampleType", values=["Calibrator", "Buffer"])
        self.assertIn("Calibrator", adat.index.get_level_values("SampleType"))
        self.assertIn("Buffer", adat.index.get_level_values("SampleType"))
        self.assertNotIn("QC", adat.index.get_level_values("SampleType"))

    def test_pick_on_columns(self):
        self.assertIn("10000-28", self.adat.columns.get_level_values("SeqId"))
        self.assertIn("10001-7", self.adat.columns.get_level_values("SeqId"))
        self.assertIn("10003-15", self.adat.columns.get_level_values("SeqId"))
        adat = self.adat.pick_on_meta(axis=1, name="SeqId", values=["10000-28", "10001-7"])
        self.assertIn("10000-28", adat.columns.get_level_values("SeqId"))
        self.assertIn("10001-7", adat.columns.get_level_values("SeqId"))
        self.assertNotIn("10003-15", adat.columns.get_level_values("SeqId"))
        self.assertEqual(len(adat.columns.get_level_values("SeqId")), 2)


class InsertIntoMetaTest(TestCase):
    def setUp(self):
        self.adat = canopy.read_adat(control_dat_path)

    def test_insert_into_row_meta(self):
        new_barcodes = list([str(i) for i in range(self.adat.shape[0])])
        self.assertNotIn("NewBarcode", self.adat.index.names)
        adat = self.adat.insert_meta(axis=0, name="NewBarcode", values=new_barcodes)
        self.assertIn("NewBarcode", adat.index.names)
        self.assertEqual(new_barcodes, list(adat.index.get_level_values("NewBarcode")))

    def test_insert_into_column_meta(self):
        new_seqids = list([str(i) for i in range(self.adat.shape[1])])
        self.assertNotIn("NewSeqIds", self.adat.columns.names)
        adat = self.adat.insert_meta(axis=1, name="NewSeqIds", values=new_seqids)
        self.assertIn("NewSeqIds", adat.columns.names)
        self.assertEqual(new_seqids, list(adat.columns.get_level_values("NewSeqIds")))


class ReplaceMetaTest(TestCase):
    def setUp(self):
        self.adat = canopy.read_adat(control_dat_path)

    def test_replace_row_meta(self):
        new_barcodes = list([str(i) for i in range(self.adat.shape[0])])
        name_order = list(self.adat.index.names)
        self.assertIn("Barcode", self.adat.index.names)
        adat = self.adat.replace_meta(axis=0, name="Barcode", values=new_barcodes)
        self.assertEqual(new_barcodes, list(adat.index.get_level_values("Barcode")))
        self.assertEqual(name_order, list(adat.index.names))

    def test_replace_column_meta(self):
        new_seqids = list([str(i) for i in range(self.adat.shape[1])])
        name_order = list(self.adat.columns.names)
        self.assertIn("SeqId", self.adat.columns.names)
        adat = self.adat.replace_meta(axis=1, name="SeqId", values=new_seqids)
        self.assertEqual(new_seqids, list(adat.columns.get_level_values("SeqId")))
        self.assertEqual(name_order, list(adat.columns.names))


class ReplaceKeyedMetaTest(TestCase):
    def setUp(self):
        rfu_data = [[1, 2, 3], [4, 5, 6]]
        col_metadata = {
            "SeqId": ["A", "B", "C"],
            "SeqIdVersion": ["1", "2", "3"],
            "ColCheck": ["PASS", "FLAG", "FLAG"],
        }
        row_metadata = {"PlateId": ["A12", "A12"], "Barcode": ["SL1234", "SL1235"]}
        header_metadata = {
            "AdatId": "1a2b3c",
            "!AssayRobot": "Tecan1, Tecan2",
            "RunNotes": "run note 1",
        }
        self.adat = Adat.from_features(rfu_data, row_metadata, col_metadata, header_metadata)

    def test_replace_keyed_row_meta(self):
        replace_values_dict = {"SL1234": "A13", "SL1235": "A13"}
        new_adat = self.adat.replace_keyed_meta(
            axis=0,
            replaced_meta_name="PlateId",
            key_meta_name="Barcode",
            values_dict=replace_values_dict,
        )
        self.assertEqual(["A13", "A13"], list(new_adat.index.get_level_values("PlateId")))

    def test_replace_keyed_column_meta(self):
        replace_values_dict = {"A": "5", "B": "6", "C": "7"}
        new_adat = self.adat.replace_keyed_meta(
            axis=1,
            replaced_meta_name="SeqIdVersion",
            key_meta_name="SeqId",
            values_dict=replace_values_dict,
        )
        self.assertEqual(["5", "6", "7"], list(new_adat.columns.get_level_values("SeqIdVersion")))


class InsertKeyedMetaTest(TestCase):
    def setUp(self):
        self.adat = canopy.read_adat(control_dat_path)

    def test_insert_row_meta(self):
        new_barcodes = {
            old_barcode: i for i, old_barcode in enumerate(self.adat.index.get_level_values("Barcode"))
        }
        self.assertNotIn("NewBarcode", self.adat.index.names)
        adat = self.adat.insert_keyed_meta(
            axis=0,
            inserted_meta_name="NewBarcode",
            key_meta_name="Barcode",
            values_dict=new_barcodes,
        )
        self.assertEqual(list(new_barcodes.values()), list(adat.index.get_level_values("NewBarcode")))

    def test_insert_column_meta(self):
        new_seqids = {
            old_seqid: i for i, old_seqid in enumerate(self.adat.columns.get_level_values("SeqId"))
        }
        self.assertNotIn("NewSeqId", self.adat.columns.names)
        adat = self.adat.insert_keyed_meta(
            axis=1,
            inserted_meta_name="NewSeqId",
            key_meta_name="SeqId",
            values_dict=new_seqids,
        )
        self.assertEqual(list(new_seqids.values()), list(adat.columns.get_level_values("NewSeqId")))


class UpdateSomamerMetadataFromAdatTestCase(TestCase):
    def setUp(self):
        rfu_data = [[1, 2, 3], [4, 5, 6]]
        col_metadata = {
            "SeqId": ["A", "B", "C"],
            "SeqIdVersion": ["1", "2", "3"],
            "ColCheck": ["PASS", "FLAG", "FLAG"],
        }
        row_metadata = {"PlateId": ["A12", "A12"], "Barcode": ["SL1234", "SL1235"]}
        header_metadata = {
            "AdatId": "1a2b3c",
            "!AssayRobot": "Tecan1, Tecan2",
            "RunNotes": "run note 1",
        }
        self.adat0 = Adat.from_features(rfu_data, row_metadata, col_metadata, header_metadata)

        rfu_data = [[5, 6, 7], [6, 5, 4]]
        col_metadata = {
            "SeqId": ["A", "B", "C"],
            "SeqIdVersion": ["4", "5", "6"],
            "ColCheck": ["PASS", "PASS", "FLAG"],
        }
        row_metadata = {"PlateId": ["A13", "A13"], "Barcode": ["SL1236", "SL1237"]}
        header_metadata = {
            "AdatId": "1a2b3d",
            "!AssayRobot": "Tecan2",
            "RunNotes": "run note 2",
        }
        self.adat1 = Adat.from_features(rfu_data, row_metadata, col_metadata, header_metadata)

    @pytest.mark.filterwarnings("ignore:Standard column")
    def test_col_meta_replaced(self):
        rfu_data = [[5, 6, 7], [6, 5, 4]]
        col_metadata = {
            "SeqId": ["A", "B", "C"],
            "SeqIdVersion": ["1", "2", "3"],
            "ColCheck": ["PASS", "PASS", "FLAG"],
        }
        row_metadata = {"PlateId": ["A13", "A13"], "Barcode": ["SL1236", "SL1237"]}
        header_metadata = {
            "AdatId": "1a2b3d",
            "!AssayRobot": "Tecan2",
            "RunNotes": "run note 2",
        }
        truth_meta_replaced_adat_1 = Adat.from_features(
            rfu_data, row_metadata, col_metadata, header_metadata
        )

        meta_replaced_adat = self.adat1.update_somamer_metadata_from_adat(self.adat0)
        pd.testing.assert_frame_equal(truth_meta_replaced_adat_1, meta_replaced_adat)

    def test_missing_col_meta_warns(self):
        with pytest.warns(
            UserWarning,
            match=r"Standard column, \w+, not found in column metadata. Continuing to next.",
        ):
            self.adat1.update_somamer_metadata_from_adat(self.adat0)

    @pytest.mark.filterwarnings("ignore:Standard column")
    def test_seq_ids_mismatch_error(self):
        rfu_data = [[5, 6, 7], [6, 5, 4]]
        col_metadata = {
            "SeqId": ["A", "B", "D"],
            "SeqIdVersion": ["4", "5", "6"],
            "ColCheck": ["PASS", "PASS", "FLAG"],
        }
        row_metadata = {"PlateId": ["A13", "A13"], "Barcode": ["SL1236", "SL1237"]}
        header_metadata = {
            "AdatId": "1a2b3d",
            "!AssayRobot": "Tecan2",
            "RunNotes": "run note 2",
        }
        mismatch_seq_id_adat = Adat.from_features(rfu_data, row_metadata, col_metadata, header_metadata)

        with self.assertRaises(AdatMetaError):
            mismatch_seq_id_adat.update_somamer_metadata_from_adat(self.adat0)
