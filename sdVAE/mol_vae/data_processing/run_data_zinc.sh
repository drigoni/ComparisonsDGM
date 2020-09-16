#!/bin/bash

root=../../..

sk=0
zinc_grammar_file=$root/sdVAE/dropbox/context_free_grammars/mol_zinc.grammar
zinc_smiles_file=zinc
zinc_save_dir=$root/sdVAE/dropbox/data/zinc

$1/python make_dataset_parallel.py \
    -grammar_file $zinc_grammar_file \
    -smiles_file $zinc_smiles_file \
    -save_dir $zinc_save_dir \
    -skip_deter $sk \
    -bondcompact 0 \
    -data_gen_threads 1

