#!/usr/bin/env python2

from __future__ import print_function

import numpy as np
from tqdm import tqdm
from rdkit import Chem

import os
import sys
sys.path.append('%s/../../../_utils' % os.path.dirname(os.path.realpath(__file__)))
from utils import save_decoded_priors, load_decoded_results

# 0. Constants
nb_latent_point = 20000
chunk_size = 100

def cal_valid_prior(model, latent_dim):
    from att_model_proxy import batch_decode
    pbar = tqdm(list(range(0, nb_latent_point, chunk_size)), desc='decoding')
    for start in pbar:
        end = min(start + chunk_size, nb_latent_point)
        latent_point = np.random.normal(size=(end - start, latent_dim))
        latent_point = latent_point.astype(np.float32)

        raw_logits = model.pred_raw_logits(latent_point, 1500)
        decoded_array = batch_decode(raw_logits, True, decode_times=1)

        decoded_priors = [[]]
        for i in range(end - start):
            s = decoded_array[i][0]
            decoded_priors[0].append(s)
    return decoded_priors

def main():
    seed = 10960817
    np.random.seed(seed)

    from att_model_proxy import AttMolProxy as ProxyModel
    from att_model_proxy import cmd_args

    model = ProxyModel()

    generation_file = cmd_args.save_dir + '/decoded_generation.txt'
    result = cal_valid_prior(model, cmd_args.latent_dim)
    save_decoded_priors(result, generation_file)




import pdb, traceback, sys, code

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
