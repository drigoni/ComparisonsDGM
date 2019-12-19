#!/bin/bash

root=../../..

zinc_grammar_file=$root/sdvae/dropbox/context_free_grammars/mol_zinc.grammar
zinc_smiles_file=$root/_dataset/ZINC/250k_rndm_zinc_drugs_clean.smi
zinc_save_dir=$root/sdvae/dropbox/data/zinc/

$1/python dump_cfg_trees.py \
    -grammar_file $zinc_grammar_file \
    -smiles_file $zinc_smiles_file \
    -save_dir $zinc_save_dir
