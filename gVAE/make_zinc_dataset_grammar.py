from __future__ import print_function
import nltk
import zinc_grammar
import numpy as np
import h5py
import molecule_vae
import sys, os
sys.path.append('%s/../_utils' % os.path.dirname(os.path.realpath(__file__)))
from read_dataset import read_zinc


L = read_zinc()

MAX_LEN=277
NCHARS = len(zinc_grammar.GCFG.productions())

def to_one_hot(smiles):
    """ Encode a list of smiles strings to one-hot vectors """
    assert type(smiles) == list
    prod_map = {}
    for ix, prod in enumerate(zinc_grammar.GCFG.productions()):
        prod_map[prod] = ix
    tokenize = molecule_vae.get_zinc_tokenizer(zinc_grammar.GCFG)
    tokens = map(tokenize, smiles)
    parser = nltk.ChartParser(zinc_grammar.GCFG)
    parse_trees = [parser.parse(t).next() for t in tokens]
    productions_seq = [tree.productions() for tree in parse_trees]
    indices = [np.array([prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
    one_hot = np.zeros((len(indices), MAX_LEN, NCHARS), dtype=np.float32)
    for i in xrange(len(indices)):
        num_productions = len(indices[i])
        one_hot[i][np.arange(num_productions),indices[i]] = 1.
        one_hot[i][np.arange(num_productions, MAX_LEN),-1] = 1.
    return one_hot


OH = np.zeros((len(L),MAX_LEN,NCHARS))
for i in range(0, len(L), 100):
    print('Processing: i=[' + str(i) + ':' + str(i+100) + ']')
    onehot = to_one_hot(L[i:i+100])
    OH[i:i+100,:,:] = onehot

h5f = h5py.File('data/zinc_grammar_dataset.h5','w')
h5f.create_dataset('data', data=OH)
h5f.close()
