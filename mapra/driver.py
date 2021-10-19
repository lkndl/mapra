#!/usr/bin/env python3

from pathlib import Path
from mapra import prep

data = prep.elaspic_dataset(Path('.').resolve().parent)
data.fetch_numpy_distances(selector={2048: True})

print('driver finished')
