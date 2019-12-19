#!/bin/bash

root=../../..

bsize=500
enc=cnn
ae_type=vae
loss_type=vanilla
rnn_type=gru
kl_coeff=1
lr=0.001
num_epochs=255
eps_std=0.01
sk=0
from joblib import Parallel, delayed

# code for both datasets
zinc_grammar_file=$root/sdVAE/dropbox/context_free_grammars/mol_zinc.grammar
zinc_save_dir=$root/sdVAE/dropbox/results/zinc


qm9_grammar_file=$root/sdVAE/dropbox/context_free_grammars/mol_qm9.grammar
qm9_save_dir=$root/sdVAE/dropbox/results/qm9

echo "save_dir for zinc use is $zinc_save_dir"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}  # only set when CUDA_VISIBLE_DEVICES is not set.
$1/python generation.py \
    -grammar_file $zinc_grammar_file \
    -batch_size $bsize \
    -skip_deter $sk \
    -bondcompact 0 \
    -encoder_type $enc \
    -save_dir $zinc_save_dir \
    -ae_type $ae_type \
    -learning_rate $lr \
    -rnn_type $rnn_type \
    -num_epochs $num_epochs \
    -eps_std $eps_std \
    -loss_type $loss_type \
    -kl_coeff $kl_coeff \
    -mode cpu \
    -saved_model $zinc_save_dir/epoch-best.model \
    $@ ;


echo "save_dir for qm9 use is $qm9_save_dir"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}  # only set when CUDA_VISIBLE_DEVICES is not set.
$1/python generation.py \
    -grammar_file $qm9_grammar_file \
    -batch_size $bsize \
    -skip_deter $sk \
    -bondcompact 0 \
    -encoder_type $enc \
    -save_dir $qm9_save_dir \
    -ae_type $ae_type \
    -learning_rate $lr \
    -rnn_type $rnn_type \
    -num_epochs $num_epochs \
    -eps_std $eps_std \
    -loss_type $loss_type \
    -kl_coeff $kl_coeff \
    -mode cpu \
    -saved_model $qm9_save_dir/epoch-best.model \
    $@ ;
