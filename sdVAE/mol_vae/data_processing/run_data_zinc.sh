#!/bin/bash

root=../../..

sk=0
zinc_grammar_file=$root/sdvae/dropbox/context_free_grammars/mol_zinc.grammar
zinc_smiles_file=$root/_dataset/ZINC/250k_rndm_zinc_drugs_clean.smi
zinc_save_dir=$root/sdvae/dropbox/data/zinc

$1/python make_dataset_parallel.py \
    -grammar_file $zinc_grammar_file \
    -smiles_file $zinc_smiles_file \
    -save_dir $zinc_save_dir \
    -skip_deter $sk \
    -bondcompact 0 \
    -data_gen_threads 1

