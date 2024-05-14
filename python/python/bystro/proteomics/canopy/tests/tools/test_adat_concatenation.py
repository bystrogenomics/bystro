from unittest import TestCase

import pytest

from bystro.proteomics.canopy import Adat
from bystro.proteomics.canopy.tools.adat_concatenation import concatenate_adats, smart_adat_concatenation
from bystro.proteomics.canopy.tools.errors import AdatConcatError


class ConcatHeadersTest(TestCase):
    def setUp(self):
        rfu_data = [[1, 2, 3], [4, 5, 6]]
        col_metadata = {'SeqId': ['A', 'B', 'C'], 'ColCheck': ['PASS', 'FLAG', 'FLAG']}
        row_metadata = {'PlateId': ['A12', 'A12'], 'Barcode': ['SL1234', 'SL1235']}
        header_metadata = {
            'AdatId': '1a2b3c',
            '!AssayRobot': 'Tecan1, Tecan2',
            'RunNotes': 'run note 1',
        }
        adat1 = Adat.from_features(
            rfu_data, row_metadata, col_metadata, header_metadata
        )

        rfu_data = [[5, 6, 7], [6, 5, 4]]
        col_metadata = {'SeqId': ['A', 'B', 'C'], 'ColCheck': ['PASS', 'PASS', 'FLAG']}
        row_metadata = {'PlateId': ['A13', 'A13'], 'Barcode': ['SL1236', 'SL1237']}
        header_metadata = {
            'AdatId': '1a2b3d',
            '!AssayRobot': 'Tecan2',
            'RunNotes': 'run note 2',
        }
        adat2 = Adat.from_features(
            rfu_data, row_metadata, col_metadata, header_metadata
        )
        self.adats = [adat1, adat2]

    def test_set_addition(self):
        concat_adat = concatenate_adats(self.adats)
        self.assertEqual(concat_adat.header_metadata['!AssayRobot'], 'Tecan1, Tecan2')

    def test_null(self):
        concat_adat = concatenate_adats(self.adats)
        self.assertIsNone(concat_adat.header_metadata['AdatId'])

    def test_str_pipe_append(self):
        concat_adat = concatenate_adats(self.adats)
        self.assertEqual(
            concat_adat.header_metadata['RunNotes'], 'run note 1 | run note 2'
        )

    def test_mismatch_raises(self):
        self.adats[0].header_metadata['CalibratorId'] = '123'
        self.adats[1].header_metadata['CalibratorId'] = '234'
        self.assertRaises(AdatConcatError, concatenate_adats, self.adats)


class ConcatRowsTest(TestCase):
    def setUp(self):
        rfu_data = [[1, 2, 3], [4, 5, 6]]
        col_metadata = {'SeqId': ['A', 'B', 'C'], 'ColCheck': ['PASS', 'FLAG', 'FLAG']}
        row_metadata = {'PlateId': ['A12', 'A12'], 'Barcode': ['SL1234', 'SL1235']}
        header_metadata = {
            'AdatId': '1a2b3c',
            '!AssayRobot': 'Tecan1, Tecan2',
            'RunNotes': 'run note 1',
        }
        adat1 = Adat.from_features(
            rfu_data, row_metadata, col_metadata, header_metadata
        )

        rfu_data = [[5, 6, 7], [6, 5, 4]]
        col_metadata = {'SeqId': ['A', 'B', 'C'], 'ColCheck': ['PASS', 'PASS', 'FLAG']}
        row_metadata = {'PlateId': ['A13', 'A13'], 'Barcode': ['SL1236', 'SL1237']}
        header_metadata = {
            'AdatId': '1a2b3d',
            '!AssayRobot': 'Tecan2',
            'RunNotes': 'run note 2',
        }
        adat2 = Adat.from_features(
            rfu_data, row_metadata, col_metadata, header_metadata
        )
        self.adats = [adat1, adat2]

    def test_row_metadata_accuracy(self):
        concat_adat = concatenate_adats(self.adats)

        barcodes = list(concat_adat.index.get_level_values('Barcode'))
        self.assertEqual(barcodes, ['SL1234', 'SL1235', 'SL1236', 'SL1237'])
        plate_ids = list(concat_adat.index.get_level_values('PlateId'))
        self.assertEqual(plate_ids, ['A12', 'A12', 'A13', 'A13'])

    def test_row_metadata_mismatch(self):
        rfu_data = [[5, 6, 7], [6, 5, 4]]
        col_metadata = {'SeqId': ['A', 'B', 'C'], 'ColCheck': ['PASS', 'PASS', 'FLAG']}
        row_metadata = {
            'PlateId': ['A13', 'A13'],
            'Barcode': ['SL1236', 'SL1237'],
            'ExtraData': ['foo', 'bar'],
        }
        header_metadata = {
            'AdatId': '1a2b3d',
            '!AssayRobot': 'Tecan2',
            'RunNotes': 'run note 2',
        }
        adat3 = Adat.from_features(
            rfu_data, row_metadata, col_metadata, header_metadata
        )

        self.adats.append(adat3)
        self.assertRaises(AdatConcatError, concatenate_adats, self.adats)


class ConcatColumnsTest(TestCase):
    def setUp(self):
        rfu_data = [[1, 2, 3], [4, 5, 6]]
        col_metadata = {'SeqId': ['A', 'B', 'C'], 'ColCheck': ['PASS', 'FLAG', 'FLAG']}
        row_metadata = {'PlateId': ['A12', 'A12'], 'Barcode': ['SL1234', 'SL1235']}
        header_metadata = {
            'AdatId': '1a2b3c',
            '!AssayRobot': 'Tecan1, Tecan2',
            'RunNotes': 'run note 1',
        }
        adat1 = Adat.from_features(
            rfu_data, row_metadata, col_metadata, header_metadata
        )

        rfu_data = [[5, 6, 7], [6, 5, 4]]
        col_metadata = {'SeqId': ['A', 'B', 'C'], 'ColCheck': ['PASS', 'PASS', 'FLAG']}
        row_metadata = {'PlateId': ['A13', 'A13'], 'Barcode': ['SL1236', 'SL1237']}
        header_metadata = {
            'AdatId': '1a2b3d',
            '!AssayRobot': 'Tecan2',
            'RunNotes': 'run note 2',
        }
        adat2 = Adat.from_features(
            rfu_data, row_metadata, col_metadata, header_metadata
        )
        self.adats = [adat1, adat2]

    def test_column_accuracy(self):
        concat_adat = concatenate_adats(self.adats)

        seq_ids = list(concat_adat.columns.get_level_values('SeqId'))
        self.assertEqual(seq_ids, ['A', 'B', 'C'])
        col_checks = list(concat_adat.columns.get_level_values('ColCheck'))
        self.assertEqual(col_checks, ['PASS', 'FLAG', 'FLAG'])

    def test_column_mismatch(self):
        rfu_data = [[5, 6, 7], [6, 5, 4]]
        col_metadata = {'SeqId': ['A', 'B', 'D'], 'ColCheck': ['PASS', 'PASS', 'FLAG']}
        row_metadata = {'PlateId': ['A13', 'A13'], 'Barcode': ['SL1236', 'SL1237']}
        header_metadata = {
            'AdatId': '1a2b3d',
            '!AssayRobot': 'Tecan2',
            'RunNotes': 'run note 2',
        }
        adat3 = Adat.from_features(
            rfu_data, row_metadata, col_metadata, header_metadata
        )
        self.adats.append(adat3)

        self.assertRaises(AdatConcatError, concatenate_adats, self.adats)


class ConcatRfuTest(TestCase):
    def setUp(self):
        rfu_data = [[1, 2, 3], [4, 5, 6]]
        col_metadata = {'SeqId': ['A', 'B', 'C'], 'ColCheck': ['PASS', 'FLAG', 'FLAG']}
        row_metadata = {'PlateId': ['A12', 'A12'], 'Barcode': ['SL1234', 'SL1235']}
        header_metadata = {
            'AdatId': '1a2b3c',
            '!AssayRobot': 'Tecan1, Tecan2',
            'RunNotes': 'run note 1',
        }
        adat1 = Adat.from_features(
            rfu_data, row_metadata, col_metadata, header_metadata
        )

        rfu_data = [[5, 6, 7], [6, 5, 4]]
        col_metadata = {'SeqId': ['A', 'B', 'C'], 'ColCheck': ['PASS', 'PASS', 'FLAG']}
        row_metadata = {'PlateId': ['A13', 'A13'], 'Barcode': ['SL1236', 'SL1237']}
        header_metadata = {
            'AdatId': '1a2b3d',
            '!AssayRobot': 'Tecan2',
            'RunNotes': 'run note 2',
        }
        adat2 = Adat.from_features(
            rfu_data, row_metadata, col_metadata, header_metadata
        )
        self.adats = [adat1, adat2]

    def test_rfu_accuracy(self):
        concat_adat = concatenate_adats(self.adats)
        self.assertTrue(
            (concat_adat.values == [[1, 2, 3], [4, 5, 6], [5, 6, 7], [6, 5, 4]]).all()
        )


class SmartConcatTestCase(TestCase):
    def setUp(self):
        rfu_data = [[1, 2, 3], [4, 5, 6]]
        col_metadata = {
            'SeqId': ['A', 'B', 'C'],
            'SeqIdVersion': ['1', '2', '3'],
            'ColCheck': ['PASS', 'FLAG', 'FLAG'],
        }
        row_metadata = {
            'PlateId': ['A12', 'A12'],
            'Barcode': ['SL1234', 'SL1235'],
            'NewColumn': ['1', '2'],
        }
        header_metadata = {
            'AdatId': '1a2b3c',
            '!AssayRobot': 'Tecan1, Tecan2',
            'RunNotes': 'run note 1',
            '!Title': 'stuff',
        }
        self.adat0 = Adat.from_features(
            rfu_data, row_metadata, col_metadata, header_metadata
        )

        rfu_data = [[5, 6, 7], [6, 5, 4]]
        col_metadata = {
            'SeqId': ['A', 'B', 'D'],
            'SeqIdVersion': ['1', '2', '3'],
            'ColCheck': ['PASS', 'PASS', 'FLAG'],
        }
        row_metadata = {
            'PlateId': ['A13', 'A13'],
            'Barcode': ['SL1236', 'SL1237'],
            'RowCheck': ['PASS', 'FLAG'],
        }
        header_metadata = {
            'AdatId': '1a2b3d',
            '!AssayRobot': 'Tecan2',
            'RunNotes': 'run note 2',
            '!Title': 'morestuff',
        }
        self.adat1 = Adat.from_features(
            rfu_data, row_metadata, col_metadata, header_metadata
        )

        rfu_data = [[8, 9, 1], [1, 4, 8]]
        col_metadata = {'SeqId': ['A', 'D', 'E'], 'SeqIdVersion': ['1', '2', '3']}
        row_metadata = {'PlateId': ['A14', 'A14'], 'Barcode': ['SL1238', 'SL1239']}
        header_metadata = {
            'AdatId': '1a2b3d',
            '!AssayRobot': 'Tecan2',
            'RunNotes': 'run note 2',
            '!Title': 'evenmorestuff',
        }
        self.adat2 = Adat.from_features(
            rfu_data, row_metadata, col_metadata, header_metadata
        )

        rfu_data = [[12, 23, 34], [45, 56, 67]]
        col_metadata = {
            'SeqId': ['A', 'B', 'C'],
            'SeqIdVersion': ['2', '3', '4'],
            'ColCheck': ['PASS', 'FLAG', 'FLAG'],
        }
        row_metadata = {'PlateId': ['A12', 'A12'], 'Barcode': ['SL1234', 'SL1235']}
        header_metadata = {
            'AdatId': '1a2b3d',
            '!AssayRobot': 'Tecan2',
            'RunNotes': 'run note 2',
            '!Title': 'whatstuff',
        }
        self.adat0a = Adat.from_features(
            rfu_data, row_metadata, col_metadata, header_metadata
        )

    @pytest.mark.filterwarnings('ignore:Removing seqIds from')
    @pytest.mark.filterwarnings('ignore:Standard column,')
    @pytest.mark.filterwarnings('ignore:Adding column to adat')
    def test_smart_adat_concat_seq_id_merge(self):
        concat_adat = smart_adat_concatenation([self.adat0, self.adat1, self.adat2])
        self.assertEqual(['A'], list(concat_adat.columns.get_level_values('SeqId')))

    @pytest.mark.filterwarnings('ignore:Removing seqIds from')
    @pytest.mark.filterwarnings('ignore:Standard column,')
    @pytest.mark.filterwarnings('ignore:Adding column to adat')
    def test_smart_adat_concat_header_meta(self):
        concat_adat = smart_adat_concatenation([self.adat0, self.adat1, self.adat2])
        expected_header = {
            'AdatId': None,
            '!AssayRobot': [
                {'adat_ids': ['stuff_A12'], 'value': 'Tecan1, Tecan2'},
                {'adat_ids': ['morestuff_A13', 'evenmorestuff_A14'], 'value': 'Tecan2'},
            ],
            'RunNotes': [
                {'adat_ids': ['stuff_A12'], 'value': 'run note 1'},
                {
                    'adat_ids': ['morestuff_A13', 'evenmorestuff_A14'],
                    'value': 'run note 2',
                },
            ],
            '!Title': [
                {'adat_ids': ['stuff_A12'], 'value': 'stuff'},
                {'adat_ids': ['morestuff_A13'], 'value': 'morestuff'},
                {'adat_ids': ['evenmorestuff_A14'], 'value': 'evenmorestuff'},
            ],
        }
        self.assertEqual(expected_header, concat_adat.header_metadata)

    @pytest.mark.filterwarnings('ignore:Removing seqIds from')
    @pytest.mark.filterwarnings('ignore:Standard column,')
    @pytest.mark.filterwarnings('ignore:Adding column to adat')
    def test_smart_adat_concat_row_meta(self):
        concat_adat = smart_adat_concatenation([self.adat0, self.adat1, self.adat2])
        expected_row_names = ['PlateId', 'Barcode', 'NewColumn', 'RowCheck']
        self.assertEqual(expected_row_names, concat_adat.index.names)

    def test_smart_adat_warnings(self):
        with pytest.warns(UserWarning) as records:
            smart_adat_concatenation([self.adat0, self.adat1, self.adat2])
        expected_warnings_regex = (
            r'(Adding column to adat: \w+)|(Removing seqIds from \w{3}: \w, \w)'
        )
        user_warnings = [rec for rec in records if rec.category == UserWarning]
        for rec in user_warnings:
            self.assertRegex(rec.message.args[0], expected_warnings_regex)
            print(rec.message.args[0])
        self.assertEqual(7, len(user_warnings))

    @pytest.mark.filterwarnings('ignore:Removing seqIds from')
    @pytest.mark.filterwarnings('ignore:Standard column,')
    @pytest.mark.filterwarnings('ignore:Adding column to adat')
    def test_smart_concat_adat_col_overwrite(self):
        concat_adat = smart_adat_concatenation([self.adat0, self.adat0a], self.adat0a)
        self.assertEqual(
            list(concat_adat.columns.get_level_values('SeqIdVersion')), ['2', '3', '4']
        )
