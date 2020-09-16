#!/bin/bash

root=../../..

zinc_grammar_file=$root/sdVAE/dropbox/context_free_grammars/mol_zinc.grammar
zinc_smiles_file=zinc
zinc_save_dir=$root/sdVAE/dropbox/data/zinc/

$1/python dump_cfg_trees.py \
    -grammar_file $zinc_grammar_file \
    -smiles_file $zinc_smiles_file \
    -save_dir $zinc_save_dir
