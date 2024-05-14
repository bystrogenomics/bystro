from bystro.proteomics.canopy.errors import AdatBaseError
import pandas as pd


def get_pd_axis(obj, axis: int) -> pd.MultiIndex:
    if axis == 0:
        return obj.index
    elif axis == 1:
        return obj.columns
    else:
        raise AdatBaseError('Not a valid axis, please choose "0" for row metadata or "1" column metadata')
   