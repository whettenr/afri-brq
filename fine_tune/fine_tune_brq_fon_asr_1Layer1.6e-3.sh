#!/bin/bash -l
#SBATCH --job-name=a1.6e-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=14:00:00
#SBATCH --mem=32G  
#SBATCH --constraint='GPURAM_Min_24GB&GPURAM_Max_32GB'





conda activate aa
cd /users/rwhetten/african_brq
train=/users/rwhetten/african_brq/fine_tune/train_with_bestrq_1layer.py
hparams=/users/rwhetten/african_brq/fine_tune/train_sb_BEST-RQ_1layer.yaml
output_folder=results/test_0.0016_epoch200_1Layer
pt_model_path=/users/rwhetten/african_brq/results/fongbe_lr_0.0016/CKPT+2025-04-11+14-37-42+00
python $train $hparams --pt_model_path $pt_model_path --output_folder $output_folder
