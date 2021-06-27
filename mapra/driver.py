#!/usr/bin/env python3

from pathlib import Path
from prep import dataset

data = dataset(Path('.').resolve().parent)

ar = data.fetch_numpy_distances()

print('driver finished')

