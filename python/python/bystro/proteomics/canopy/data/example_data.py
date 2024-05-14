import os

import canopy

path = os.path.dirname(canopy.__file__)

example_data = canopy.read_adat(f'{path}/data/example_data.adat')
