#!/bin/bash

root=../../..

qm9_grammar_file=$root/sdVAE/dropbox/context_free_grammars/mol_qm9.grammar
qm9_smiles_file=qm9
qm9_save_dir=$root/sdVAE/dropbox/data/qm9/


$1/python dump_cfg_trees.py \
    -grammar_file $qm9_grammar_file \
    -smiles_file $qm9_smiles_file \
    -save_dir $qm9_save_dir
