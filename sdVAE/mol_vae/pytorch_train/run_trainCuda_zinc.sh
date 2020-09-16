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

zinc_grammar_file=$root/sdVAE/dropbox/context_free_grammars/mol_zinc.grammar
zinc_data_dump=$root/sdVAE/dropbox/data/zinc/zinc-${sk}.h5
zinc_save_dir=$root/sdVAE/dropbox/results/graph_generation/zinc_cuda/vanilla-sk-${sk}-mol_${ae_type}/enc-${enc}-loss-${loss_type}-eps-${eps_std}-rnn-${rnn_type}-kl-${kl_coeff}


if [ ! -e $zinc_save_dir ];
then
    mkdir -p $zinc_save_dir
fi

export CUDA_VISIBLE_DEVICES=0
$1/python trainCuda_zinc.py \
    -grammar_file $zinc_grammar_file \
    -data_dump $zinc_data_dump \
    -old $old \
    -batch_size $bsize \
    -encoder_type $enc \
    -save_dir $zinc_save_dir \
    -ae_type $ae_type \
    -learning_rate $lr \
    -rnn_type $rnn_type \
    -num_epochs $num_epochs \
    -eps_std $eps_std \
    -loss_type $loss_type \
    -kl_coeff $kl_coeff \
    -prob_fix $prob_fix \
    -mode gpu

