from pathlib import Path
from zipfile import ZipFile

import pandas as pd

zip_path = Path(__file__).parent / 'lift.zip'


def check_substrings(full, sub1, sub2):
    if not sub1 in full:
        return False
    if not sub2 in full:
        return False
    return full.index(sub1) < full.index(sub2)


def getSomaScanLiftCCC():
    """Return the SomaScan Lifting Lin's CCC DataFrame."""
    data = []
    versions = ['s11k.json', 's7k.json', 's5k.json']
    with ZipFile(zip_path, 'r') as zp:
        for version in versions:
            df = pd.read_json(zp.open(version))
            data.append(df)
    data = pd.concat(data, axis=1).T
    data = data.drop_duplicates().T
    cols = [x for x in data.columns if "Lin's CCC" in x]
    ccc = data.loc[:, cols]
    return ccc


class LiftData:
    """Lift ADAT data in RFU space from one assay version to another using built-in references"""

    _supported_matrices = {'plasma', 'serum'}
    _version_map = {'v5.0': 's11k.json', 'v4.1': 's7k.json', 'v4.0': 's5k.json'}
    _zip_path = zip_path

    def __init__(self, from_plex, to_plex, matrix):
        """Instantiate a LiftData Object.

        Parameters:
        from_plex: The SomaScan assay version to lift from i.e. v5.0.
        to_plex: The SomaScan assay version to lift to ie i.e. v4.1
        matrix: The matrix you would like a reference for. 'serum' and 'plasma' are supported.
        """
        # instantiate these variables they should not persist accross the class.
        self._scale_factors = pd.Series(dtype='float')
        self._matrix = None
        self._lins_ccc = pd.Series(dtype='float')
        # assign the user values
        self.from_plex = from_plex
        self.to_plex = to_plex
        self.matrix = matrix
        self._df = self._read_zip(self.from_plex)

    def _read_zip(self, from_plex):
        """Load the zipped subfolders and extract it into memory."""
        with ZipFile(self._zip_path, 'r') as zp:
            df = pd.read_json(zp.open(self._version_map[from_plex]))
            # missing values are nan
            df.fillna(1.0, inplace=True)
        return df

    def _get_colname(self, kind='Scalar'):
        """Iterate through the column names and find the one that matches the __init__ parameters and kind.

        Parameters:
        kind: str. 'Scalar' or 'CCC' along with the assay versions determines the column name returned.

        Returns:
        col: str. A column name from the reference data.
        """
        mat = self.matrix.capitalize()
        for col in self._df.columns:
            tests = (
                mat in col,
                kind in col,
                check_substrings(col, self.from_plex, self.to_plex),
            )
            if all(tests):
                return col
        if self.from_plex == self.to_plex:
            raise ValueError(
                f'No lift is needed from {self.from_plex} to {self.to_plex}'
            )
        raise ValueError(
            f'Unable to match column names with {(mat, self.from_plex, self.to_plex)}.'
        )

    def _extract_reference(self):
        """From the reference DataFrame scalars and CCC for the target matrix and target version."""
        self._scale_factors = self._df[self._get_colname(kind='Scalar')]
        self._lins_ccc = self._df[self._get_colname(kind='CCC')]

    @property
    def scale_factors(self):
        """Lazy load scale factors."""
        if self._scale_factors.empty:
            self._extract_reference()
        return self._scale_factors

    @scale_factors.setter
    def scale_factors(self, scale_factors):
        if isinstance(scale_factors, pd.Series):
            self._scale_factors = scale_factors
        else:
            raise TypeError(
                'LiftData.scale_factors must be a pandas.Series with data type float'
            )

    @property
    def lins_ccc(self):
        if self._lins_ccc.empty:
            """Lazy load Lin's CCC."""
            self._extract_reference()
        return self._lins_ccc

    @property
    def from_plex(self):
        return self._from_plex

    @from_plex.setter
    def from_plex(self, from_plex: str):
        # don't let them use a non-string
        if not isinstance(from_plex, str):
            raise TypeError('from_plex expects a string ie "v5.0"')
        # always lower case.
        from_plex = from_plex.lower()
        # make sure we can interpret it.
        if not from_plex in self._version_map.keys():
            raise ValueError(
                f'from_plex "{from_plex}" must be one of the supported assay versions {[x for x in self._version_map.keys()]}'
            )
        self._from_plex = from_plex

    @property
    def to_plex(self):
        return self._to_plex

    @to_plex.setter
    def to_plex(self, to_plex: str):
        # don't let them use a non-string
        if not isinstance(to_plex, str):
            raise TypeError('to_plex expects a string ie "v5.0"')
        # always lower case.
        to_plex = to_plex.lower()
        # make sure we can interpret it.
        if not to_plex in self._version_map.keys():
            raise ValueError(
                f'to_plex "{to_plex}" must be one of the supported assay versions {[x for x in self._version_map.keys()]}'
            )
        self._to_plex = to_plex

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: str):
        if not isinstance(matrix, str):
            raise TypeError(
                f"Matrix must be a string. Supported matrices are: {self._supported_matrices}"
            )
        matrix = matrix.lower()
        if matrix in self._supported_matrices:
            self._matrix = matrix
        else:
            raise ValueError(
                f'"{matrix}" is not a supported matrix. Supported matrices are: {self._supported_matrices}'
            )
