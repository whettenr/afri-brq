#!/bin/bash -l
#SBATCH --job-name=ASR
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=14:00:00
#SBATCH --mem=32G  
#SBATCH --constraint='GPURAM_Min_24GB&GPURAM_Max_32GB'



conda activate aa
cd /users/rwhetten/african_brq
train=/users/rwhetten/african_brq/fine_tune/train_with_bestrq_noaug.py
hparams=/users/rwhetten/african_brq/fine_tune/train_sb_BEST-RQ.yaml
output_folder=results/test_all_3_0.0016__exclude_2Lin_noaug
pt_model_path=/users/rwhetten/african_brq/results/all_3_lr_0.0016/save/CKPT+2025-04-16+02-07-05+00

python $train $hparams --pt_model_path $pt_model_path --output_folder $output_folder


