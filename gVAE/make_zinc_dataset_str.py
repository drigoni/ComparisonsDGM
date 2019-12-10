import numpy as np
from models.utils import many_one_hot
import h5py

import sys, os
sys.path.append('%s/../_utils' % os.path.dirname(os.path.realpath(__file__)))
from read_dataset import read_zinc

L = read_zinc()

chars = ['C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[', '@', 'H', ']', 'n', '-', '#', 'S', 'l', '+', 's', 'B', 'r', '/', '4', '\\', '5', '6', '7', 'I', 'P', '8', ' ']
DIM = len(chars)


count = 0
MAX_LEN = 120
OH = np.zeros((249456,MAX_LEN,DIM))
for chem in L:
    indices = []
    for c in chem:
        indices.append(chars.index(c))
    if len(indices) < MAX_LEN:
        indices.extend((MAX_LEN-len(indices))*[DIM-1])
    OH[count,:,:] = many_one_hot(np.array(indices), DIM)
    count = count + 1
h5f = h5py.File('data/zinc_str_dataset.h5','w')
h5f.create_dataset('data', data=OH)
h5f.create_dataset('chr',  data=chars)
h5f.close()
