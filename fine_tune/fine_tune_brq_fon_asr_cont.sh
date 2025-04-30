#!/bin/bash -l
#SBATCH --job-name=ASR
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=14:00:00
#SBATCH --mem=32G  
#SBATCH --constraint='GPURAM_Min_40GB'




conda activate aa
cd /users/rwhetten/african_brq
train=/users/rwhetten/african_brq/fine_tune/train_with_bestrq.py
hparams=/users/rwhetten/african_brq/fine_tune/train_sb_BEST-RQ.yaml
output_folder=results/test_continual_pretraining_5k
pt_model_path=/users/rwhetten/african_brq/results/continual_pretrainin_fongbe/save/CKPT+2025-04-19+16-16-47+00

python $train $hparams --pt_model_path $pt_model_path --output_folder $output_folder \
    --pt_model_output_dim 1024 \
    --num_encoder_layers 24 \
    --d_ffn 4096 \
    --attention_type RelPosMHAXL \
    --batch_size 2 \
    --test_batch_size 3 \
    --grad_accumulation_factor 3 \
    --num_workers 2 \
    --number_of_epochs 10
