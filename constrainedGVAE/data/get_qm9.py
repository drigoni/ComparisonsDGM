#!/usr/bin/env/python
"""
Usage:
    get_qm9.py

Options:
    -h --help                Show this screen.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import QED
import glob
import json
import numpy as np
from utils import bond_dict, dataset_info, need_kekulize, to_graph, graph_to_adj_mat
import utils
import pickle
import random
from docopt import docopt

sys.path.append('%s/../../_utils' % os.path.dirname(os.path.realpath(__file__)))
from read_dataset import readStr_qm9

dataset = 'qm9'

def get_validation_file_names(unzip_path):
    print('loading train/validation split')
    with open('valid_idx_qm9.json', 'r') as f:
        valid_idx = json.load(f)['valid_idxs']
    valid_files = [os.path.join(unzip_path, 'dsgdb9nsd_%s.xyz' % i) for i in valid_idx]
    return valid_files

def read_xyz(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        smiles = lines[-2].split('\t')[0]
        mu = QED.qed(Chem.MolFromSmiles(smiles))
    return {'smiles': smiles, 'QED': mu}   

def train_valid_split(unzip_path):
    print('reading data...')
    raw_data = {'train': [], 'valid': []} # save the train, valid dataset.
    all_files = glob.glob(os.path.join(unzip_path, '*.xyz'))
    valid_files = get_validation_file_names(unzip_path)
    
    file_count = 0
    for file_idx, file_path in enumerate(all_files):
        if file_path not in valid_files:
            raw_data['train'].append(read_xyz(file_path))
        else:
            raw_data['valid'].append(read_xyz(file_path))
        file_count += 1
        if file_count % 2000 == 0:
            print('finished reading: %d' % file_count, end='\r')
    return raw_data

def preprocess(raw_data, dataset):
    print('parsing smiles as graphs...')
    processed_data = {'train': [], 'valid': []}
    
    file_count = 0
    for section in ['train', 'valid']:
        all_smiles = [] # record all smiles in training dataset
        for i,(smiles, QED) in enumerate([(mol['smiles'], mol['QED']) 
                                          for mol in raw_data[section]]):
            nodes, edges = to_graph(smiles, dataset)
            if len(edges) <= 0:
                continue
            processed_data[section].append({
                'targets': [[(QED)]],
                'graph': edges,
                'node_features': nodes,
                'smiles': smiles
            })
            all_smiles.append(smiles)
            if file_count % 2000 == 0:
                print('finished processing: %d' % file_count, end='\r')
            file_count += 1
        print('%s: 100 %%      ' % (section))
        # save the dataset
        with open('molecules_%s_%s.json' % (section, dataset), 'w') as f:
            json.dump(processed_data[section], f)
        # save all molecules in the training dataset
        if section == 'train':
            utils.dump('smiles_%s.pkl' % dataset, all_smiles)

    print(len(processed_data['train']) + len(processed_data['valid']))




def check_drigoni(download_path):

    def conv(smile):
        return Chem.MolToSmiles(Chem.MolFromSmiles(smile))

    print('reading data...')
    raw_data = train_valid_split(download_path)
    data = [conv(data_item['smiles']) for data_item in raw_data['valid']]
    data.extend([conv(data_item['smiles']) for data_item in raw_data['train']])


    print('reading dataset')
    dataset = readStr_qm9()

    print('len dataset:' + str(len(dataset)) + "  len data: " + str(len(data)))

    count = 0
    saw = 0
    for i, data_item in enumerate(data):
        if data_item not in dataset:
            count +=1
        saw += 1
    print('end: ' + str(count) + " saw = " + str(saw))


    count = 0
    saw = 0
    for i, data_item in enumerate(dataset):
        if data_item not in data:
            count +=1
        saw += 1
    print('end: ' + str(count) + " saw = " + str(saw))


def train_valid_split_drigoni():

    print('reading dataset')
    dataset = readStr_qm9()[5000:]

    valid_idx = np.random.randint(0, high=len(dataset), size=round(len(dataset)*0.1))

    raw_data = {'train': [], 'valid': []}  # save the train, valid dataset.
    file_count=0
    for i, smiles in enumerate(dataset):
        val = QED.qed(Chem.MolFromSmiles(smiles))
        if i not in valid_idx:
            raw_data['train'].append({'smiles': smiles, 'QED': val})
        else:
            raw_data['valid'].append({'smiles': smiles, 'QED': val})
        file_count += 1
        if file_count % 2000 == 0:
            print('finished reading: %d' % file_count, end='\r')
    return raw_data



if __name__ == "__main__":
    # download   
    #download_path = 'dsgdb9nsd.xyz.tar.bz2'
    #if not os.path.exists(download_path):
    #    print('downloading data to %s ...' % download_path)
    #    source = 'https://ndownloader.figshare.com/files/3195389'
    #    os.system('wget -O %s %s' % (download_path, source))
    #    print('finished downloading')
        
    # unzip
    #unzip_path = 'qm9_raw'
    #if not os.path.exists(unzip_path):
    #    print('extracting data to %s ...' % unzip_path)
    #    os.mkdir(unzip_path)
    #    os.system('tar xvjf %s -C %s' % (download_path, unzip_path))
    #    print('finished extracting')


    raw_data = train_valid_split_drigoni()
    #raw_data = train_valid_split(unzip_path)
    preprocess(raw_data, dataset)
