#!/usr/bin/env python2

from __future__ import print_function

import pdb, traceback, sys, code
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

import os
import sys
sys.path.append('%s/../../../_utils' % os.path.dirname(os.path.realpath(__file__)))
from read_dataset import read_zinc, readStr_qm9
from utils import save_decoded_results


# 0. Constants
nb_smiles = 5000
chunk_size = 100
encode_times = 20
decode_times = 1


def reconstruct_single(model, smiles):
    print('a chunk starts...')
    decode_result = []

    chunk = smiles
    chunk_result = [[] for _ in range(len(chunk))]
    for _encode in range(encode_times):
        z1 = model.encode(chunk, use_random=True)
        this_encode = []
        encode_id, encode_total = _encode + 1, encode_times
        for _decode in tqdm(list(range(decode_times)),
                'encode %d/%d decode' % (encode_id, encode_total)
            ):
            _result = model.decode(z1, use_random=True)
            for index, s in enumerate(_result):
                chunk_result[index].append(s)

    decode_result.extend(chunk_result)
    assert len(decode_result) == len(smiles)
    return decode_result

def reconstruct(model, smiles):
    chunk_result = Parallel(n_jobs=1)(
        delayed(reconstruct_single)(model, smiles[chunk_start: chunk_start + chunk_size])
        for chunk_start in range(0, len(smiles), chunk_size)
    )
    # '''

    decode_result = [_1 for _0 in chunk_result for _1 in _0]
    assert len(decode_result) == len(smiles)
    return decode_result


def main():
    from att_model_proxy import AttMolProxy as ProxyModel
    from att_model_proxy import cmd_args
    # takes the model and calculate the decode results
    model = ProxyModel()
    # update where to save
    decoded_file = cmd_args.save_dir + '/decoded_results.txt'



    # reading smiles test set
    if cmd_args.smiles_file == 'qm9':
        smiles_list = readStr_qm9()
    elif cmd_args.smiles_file == 'zinc':
        smiles_list = read_zinc()

    XTE = smiles_list[0:nb_smiles]

    decoded_result = reconstruct(model, XTE)
    decoded_result = np.array(decoded_result)
    save_decoded_results(XTE, decoded_result, decoded_file)


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
