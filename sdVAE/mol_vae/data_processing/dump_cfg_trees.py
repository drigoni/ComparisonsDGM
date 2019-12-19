#!/usr/bin/env python

from __future__ import print_function
from past.builtins import range

import os
import sys
import numpy as np
import math
import random

import torch
from torch.autograd import Variable

from joblib import Parallel, delayed
from rdkit import Chem

sys.path.append('%s/../mol_common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args
from mol_tree import AnnotatedTree2MolTree, get_smiles_from_tree, Node

sys.path.append('%s/../mol_vae' % os.path.dirname(os.path.realpath(__file__)))
from mol_vae import MolVAE, MolAutoEncoder

sys.path.append('%s/../mol_decoder' % os.path.dirname(os.path.realpath(__file__)))
from attribute_tree_decoder import create_tree_decoder
from mol_decoder import batch_make_att_masks
from tree_walker import OnehotBuilder, ConditionalDecoder

sys.path.append('%s/../cfg_parser' % os.path.dirname(os.path.realpath(__file__)))
import cfg_parser as parser

sys.path.append('%s/../../../_utils' % os.path.dirname(os.path.realpath(__file__)))
from read_dataset import read_qm9
from read_dataset import read_zinc



def parse_single(smiles, grammar):
    ts = parser.parse(smiles, grammar)
    assert isinstance(ts, list) and len(ts) == 1
    n = AnnotatedTree2MolTree(ts[0])
    return n

def parse_many(chunk, grammar):
    return [parse_single(smiles, grammar) for smiles in chunk]

def parse(chunk, grammar):
    size = 100
    result_list = Parallel(n_jobs=-1)(delayed(parse_many)(chunk[i: i + size], grammar) for i in range(0, len(chunk), size))
    return [_1 for _0 in result_list for _1 in _0]

import cPickle as cp

from tqdm import tqdm

if __name__ == '__main__':
    smiles_file = cmd_args.smiles_file
    save_dir = cmd_args.save_dir
    fname = save_dir + (smiles_file.split('/')[-1]).split('.')[0] + '.cfg_dump'
    fout = open(fname, 'wb')
    grammar = parser.Grammar(cmd_args.grammar_file)

    smiles = []
    if cmd_args.smiles_file.endswith('.sdf'):
        D = read_qm9()
        # fix problem about molecule with '.' inside
        for mol in D:
            if "." not in mol:
                smiles.append(mol)
    elif cmd_args.smiles_file.endswith('.smi'):
        smiles = read_zinc()
    
    for i in tqdm(range(len(smiles))):
        ts = parser.parse(smiles[i], grammar)
        assert isinstance(ts, list) and len(ts) == 1
        n = AnnotatedTree2MolTree(ts[0])
        cp.dump(n, fout, cp.HIGHEST_PROTOCOL)

    fout.close()
