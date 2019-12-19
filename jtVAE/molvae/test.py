import torch
import torch.nn as nn
from torch.autograd import Variable

import math, random, sys
from optparse import OptionParser
from collections import deque

import rdkit
import rdkit.Chem as Chem

import sys
import os

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from jtnn import *

sys.path.append('%s/../../_utils' % os.path.dirname(os.path.realpath(__file__)))
from read_dataset import readStr_qm9
from read_dataset import read_zinc
from utils import save_decoded_results, save_decoded_priors

import numpy as np

# constant
CHUNK_SIZE = 1

def reconstruction(model, XTE, ne, nd):
    print('Start reconstruction')
    # for every row (smile), it contains all the decoded smiles (smiles). len(XTE)*(ENCODE_TIMES*DECODE_TIMES)
    decode_result = []

    # for every group of smile
    for start_chunk in range(0, len(XTE)):
        smiles = XTE[start_chunk]

        dec_smiles_witherror = model.recon_eval(smiles, ne, nd)
        res_clean = sanitize(dec_smiles_witherror)
        decode_result.append(res_clean)

    assert len(decode_result) == len(XTE)
    return np.array(decode_result)

def sanitize(smiles):
    res = []
    for smile in smiles:
        if isinstance(smile, basestring):
            res.append(smile)
        else:
            res.append("error")
    return res

def main():
    torch.manual_seed(0)
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = OptionParser()
    parser.add_option("-t", "--test", dest="test_path")
    parser.add_option("-v", "--vocab", dest="vocab_path")
    parser.add_option("-m", "--model", dest="model_path")
    parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
    parser.add_option("-l", "--latent", dest="latent_size", default=56)
    parser.add_option("-d", "--depth", dest="depth", default=3)
    opts,args = parser.parse_args()

    vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)]
    vocab = Vocab(vocab)

    hidden_size = int(opts.hidden_size)
    latent_size = int(opts.latent_size)
    depth = int(opts.depth)

    model = JTNNVAE(vocab, hidden_size, latent_size, depth)
    model.load_state_dict(torch.load(opts.model_path))
    model = model.cpu()

    dataset_name = opts.test_path
    result_file = dataset_name + "_decoded_results.txt"
    priors_file = dataset_name + "_decoded_priors.txt"
    generation_fie = dataset_name + "_generation.txt"

    # read dataset
    if dataset_name == "zinc":
       XTE = read_zinc()
    else:
       D = readStr_qm9()
        # fix problem about molecule with '.' inside
       XTE = []
       for mol in D:
           if "." not in mol:
               XTE.append(mol)

    # reconstruction
    XTE = XTE[0:5000]
    XTE = filter(lambda x: len(x) > 1, XTE)  #needed for removing smiles with only a char.
    decoded_result = reconstruction(model, XTE, 20, 1)
    save_decoded_results(XTE, decoded_result, result_file)

    # prior
    # decoded_priors_witherrors = model.sample_prior_eval(True, 1000, 10)
    # decoded_priors = []
    # for i in decoded_priors_witherrors:
    #     decoded_priors.append(sanitize(i))
    # save_decoded_priors(decoded_priors, priors_file)

    # generation
    generation_witherrors = model.sample_prior_eval(True, 20000, 1)
    generation = []
    for i in generation_witherrors:
        generation.append(sanitize(i))
    save_decoded_priors(generation, generation_fie)






if __name__ == '__main__':
    main()
