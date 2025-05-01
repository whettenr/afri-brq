#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=100:00:00
#SBATCH --job-name=e
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=2
# SBATCH --nodelist=hemera
#SBATCH --constraint='GPURAM_Min_24GB&GPURAM_Max_32GB'
#SBATCH --mail-type=END,FAIL


conda activate aa
cd /users/rwhetten/african_brq
train=train/train.py
hparams=hparams/BEST-RQ.yaml

lr=0.0004
output_folder=results/eng/eng_noaug
train_csv=store/tgts_train.csv
valid_csv=store/tgts_dev-clean.csv

python -m torch.distributed.run --nproc_per_node=2 --rdzv_backend c10d --rdzv-endpoint=localhost:0 $train $hparams --find_unused_parameters \
    --grad_accumulation_factor 12 --output_folder $output_folder \
    --skip_prep true --lr $lr  --number_of_epochs 100 --precision fp16 \
    --train_csv $train_csv --valid_csv $valid_csv


