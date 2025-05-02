#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=100:00:00
#SBATCH --job-name=efanm
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=2
# SBATCH --nodelist=hemera
#SBATCH --constraint='GPURAM_Min_24GB&GPURAM_Max_32GB'
#SBATCH --mail-type=END,FAIL


conda activate aa
cd /users/rwhetten/african_brq
train=train/train.py
hparams=hparams/BEST-RQ-aug-nr.yaml

lr=0.0008
output_folder=results/efiy/efiy_aug_nsm_${lr}
train_csv=store/tgts_efiy.csv
valid_csv=store/tgts_valid.csv

python -m torch.distributed.run --nproc_per_node=1 --rdzv_backend c10d --rdzv-endpoint=localhost:0 $train $hparams --find_unused_parameters \
    --grad_accumulation_factor 12 --output_folder $output_folder \
    --skip_prep true --lr $lr  --number_of_epochs 600 --precision fp16 --enable_add_music True \
    --train_csv $train_csv --valid_csv $valid_csv    


