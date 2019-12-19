#!/bin/bash

root=../../..

sk=0
bsize=300
enc=cnn
ae_type=vae
loss_type=vanilla
rnn_type=gru
kl_coeff=1
lr=0.0001
num_epochs=500
eps_std=0.01
prob_fix=0

qm9_grammar_file=$root/sdvae/dropbox/context_free_grammars/mol_qm9.grammar
qm9_data_dump=$root/sdvae/dropbox/data/qm9/gdb9-${sk}.h5
qm9_save_dir=$root/sdvae/dropbox/results/graph_generation/qm9/vanilla-sk-${sk}-mol_${ae_type}/enc-${enc}-loss-${loss_type}-eps-${eps_std}-rnn-${rnn_type}-kl-${kl_coeff}


if [ ! -e $qm9_save_dir ];
then
    mkdir -p $qm9_save_dir
fi

export CUDA_VISIBLE_DEVICES=0
$1/python train_zinc.py \
    -grammar_file $qm9_grammar_file \
    -data_dump $qm9_data_dump \
    -old $old \
    -batch_size $bsize \
    -encoder_type $enc \
    -save_dir $qm9_save_dir \
    -ae_type $ae_type \
    -learning_rate $lr \
    -rnn_type $rnn_type \
    -num_epochs $num_epochs \
    -eps_std $eps_std \
    -loss_type $loss_type \
    -kl_coeff $kl_coeff \
    -prob_fix $prob_fix \
    -mode cpu
