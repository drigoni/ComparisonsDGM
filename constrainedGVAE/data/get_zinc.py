#!/usr/bin/env/python
"""
Usage:
    get_data.py --dataset zinc|qm9|cep

Options:
    -h --help                Show this screen.
    --dataset NAME           Dataset name: zinc, qm9, cep
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import QED
import glob
import csv, json
import numpy as np
from utils import bond_dict, dataset_info, need_kekulize, to_graph,graph_to_adj_mat
import utils
import pickle
import random
from docopt import docopt
from get_qm9 import preprocess

sys.path.append('%s/../../_utils' % os.path.dirname(os.path.realpath(__file__)))
from read_dataset import read_zinc

dataset = "zinc"

def train_valid_split(download_path):
    # load validation dataset
    with open("valid_idx_zinc.json", 'r') as f:
        valid_idx = json.load(f)

    print('reading data...')
    raw_data = {'train': [], 'valid': []} # save the train, valid dataset.
    with open(download_path, 'r') as f:
        all_data = list(csv.DictReader(f))

    file_count=0
    for i, data_item in enumerate(all_data):
        smiles = data_item['smiles'].strip()
        QED = float(data_item['qed'])
        if i not in valid_idx:
            raw_data['train'].append({'smiles': smiles, 'QED': QED})
        else:
            raw_data['valid'].append({'smiles': smiles, 'QED': QED})
        file_count += 1
        if file_count % 2000 ==0:
            print('finished reading: %d' % file_count, end='\r')
    return raw_data


def train_valid_split_drigoni(download_path):
    # load validation dataset
    with open("valid_idx_zinc.json", 'r') as f:
        valid_idx = json.load(f)

    print("n Valid: " + str(len(valid_idx)))

    print('reading data...')
    raw_data = {'train': [], 'valid': []} # save the train, valid dataset.
    with open(download_path, 'r') as f:
        all_data = list(csv.DictReader(f))

    file_count=0
    for i, data_item in enumerate(all_data):
        smiles = data_item['smiles'].strip()
        QED = float(data_item['qed'])
        if i > 4999 and i not in valid_idx:
            raw_data['train'].append({'smiles': smiles, 'QED': QED})
        elif i > 4999:
            raw_data['valid'].append({'smiles': smiles, 'QED': QED})
        file_count += 1
        if file_count % 2000 ==0:
            print('finished reading: %d' % file_count, end='\r')
    return raw_data



    print('end: ' + str(count) + " saw = " + str(saw))

            
if __name__ == "__main__":
    download_path = '250k_rndm_zinc_drugs_clean_3.csv'
    if not os.path.exists(download_path):
        print('downloading data to %s ...' % download_path)
        source = 'https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv'
        os.system('wget -O %s %s' % (download_path, source))
        print('finished downloading')

    #needed to do the same validation of others
    raw_data = train_valid_split_drigoni(download_path)
    #raw_data = train_valid_split_(download_path)
    preprocess(raw_data, dataset)
