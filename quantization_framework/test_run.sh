#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --job-name=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=2
#SBATCH --nodelist=hemera
#SBATCH --constraint='GPURAM_Min_24GB&GPURAM_Max_32GB'
#SBATCH --mail-type=END,FAIL


conda activate aa
cd /users/rwhetten/african_brq/quantization_framework
ckpt_name=/users/rwhetten/african_brq/results/fongbe_lr_0.0016/CKPT+2025-04-11+14-37-42+00
train_csv=/users/rwhetten/african_brq/csvs/cappfm.csv
valid_csv=/users/rwhetten/african_brq/csvs/valid.csv
quantize_layer=6

python quantize_from_layer_GPU.py BEST-RQ.yaml \
    --train_csv $train_csv \
    --valid_csv $valid_csv \
    --precision fp16 \
    --save_folder /users/rwhetten/african_brq/results/fongbe_lr_0.0016/ \
    --ckpt_name $ckpt_name \
    --quantize_layer $quantize_layer \
    --num_encoder_layers 12 \
    --output_hidden_states True \
    --attention_type RoPEMHA \
    --save_targets_folder /users/rwhetten/african_brq/quantization_framework/targets \
    --seconds_per_batch 300 \
    --train_num_buckets 70 \
    --lq_vocab 1024




conda activate aa
cd /users/rwhetten/african_brq/quantization_framework
ckpt_name=/users/rwhetten/african_brq/results/fongbe_lr_0.0016/CKPT+2025-04-11+14-37-42+00
train_csv=/users/rwhetten/african_brq/csvs/cappfm.csv
valid_csv=/users/rwhetten/african_brq/csvs/valid.csv
quantize_layer=6

python get_features_from_layer_GPU.py BEST-RQ.yaml \
    --train_csv $train_csv \
    --valid_csv $valid_csv \
    --precision fp16 \
    --save_folder /users/rwhetten/african_brq/results/fongbe_lr_0.0016/ \
    --ckpt_name $ckpt_name \
    --quantize_layer $quantize_layer \
    --num_encoder_layers 12 \
    --output_hidden_states True \
    --attention_type RoPEMHA \
    --save_targets_folder /users/rwhetten/african_brq/quantization_framework/targets \
    --seconds_per_batch 300 \
    --train_num_buckets 70 \
    --lq_vocab 1024 \
    --save_embeddings_folder /users/rwhetten/african_brq/quantization_framework/embeddings

