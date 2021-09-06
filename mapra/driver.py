#!/usr/bin/env python3

from pathlib import Path
from prep import dataset

data = dataset(Path('.').resolve().parent)

ar = data.fetch_numpy_distances(test_set=data.uniprot_train_test_split(test_size=.1))

print('driver finished')
