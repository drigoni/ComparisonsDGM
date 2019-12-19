#!/bin/bash

root=../../..

qm9_grammar_file=$root/sdvae/dropbox/context_free_grammars/mol_qm9.grammar
qm9_smiles_file=$root/_dataset/QM9/gdb9.sdf
qm9_save_dir=$root/sdvae/dropbox/data/qm9/


$1/python dump_cfg_trees.py \
    -grammar_file $qm9_grammar_file \
    -smiles_file $qm9_smiles_file \
    -save_dir $qm9_save_dir
