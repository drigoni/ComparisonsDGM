#!/bin/bash

root=../../..

sk=0



qm9_grammar_file=$root/sdVAE/dropbox/context_free_grammars/mol_qm9.grammar
qm9_smiles_file=qm9
qm9_save_dir=$root/sdVAE/dropbox/data/qm9

$1/python make_dataset_parallel.py \
    -grammar_file $qm9_grammar_file \
    -smiles_file $qm9_smiles_file \
    -save_dir $qm9_save_dir \
    -skip_deter $sk \
    -bondcompact 0 \
    -data_gen_threads 1
