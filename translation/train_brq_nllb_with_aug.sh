#!/bin/bash -l
#SBATCH --job-name=brqnllb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G  
#SBATCH --constraint='GPURAM_Min_24GB&GPURAM_Max_32GB'


cd /users/rwhetten/african_brq
conda activate aa 
train=/users/rwhetten/african_brq/translation/train_brq_nllb.py
yaml=/users/rwhetten/african_brq/translation/train_brq_nllb_with_aug.yaml
brq_path=/users/rwhetten/african_brq/results/fongbe_lr_0.0016_ct_200e_aug/CKPT-250
python $train $yaml --brq_path $brq_path