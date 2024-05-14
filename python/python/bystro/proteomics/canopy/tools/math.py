import numpy as np


def jround(x: float, n_digits: int = 1) -> float:
    """Given a numerical value returns a rounded float.

    Is consistent with Java's default rounding utility.
    Created by Eduardo.

    Parameters
    ----------
    x : float
        Input numerical value

    n_digits : int
        Rounds x to n_digits digits. Default = 1

    Examples
    --------
    >>> x_rounded = jround(x, 3)

    Returns
    -------
    x_rounded : float
        x rounded to n_digit digits consistent with Java's rounding utility

    """
    dig = np.floor(n_digits + 0.5)
    sgn = 1
    if(x < 0.):
        sgn = -1.
        x = -x

    if (dig == 0):
        return(sgn * np.rint(x))
    elif (dig > 0):
        pow10 = pow(10, dig)
        intx = np.floor(x)
        return(sgn * (intx + np.rint((x - intx) * pow10) / pow10))
    else:
        pow10 = pow(10, dig)
        return(sgn * np.rint(x / pow10) * pow10)
