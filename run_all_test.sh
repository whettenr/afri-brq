#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=i❤️🌍
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=2
# SBATCH --nodelist=hemera
#SBATCH --constraint='GPURAM_Min_24GB&GPURAM_Max_32GB'
#SBATCH --mail-type=END,FAIL


conda activate aa
cd /users/rwhetten/african_brq

train=train.py
hparams=BEST-RQ.yaml
data_folder=/users/fkponou/data/speechbrain/To_Ryan
train_csv=/users/rwhetten/african_brq/csvs/f_i_y.csv

lr=0.0008
output_folder=results/all_lr_${lr}

python -m torch.distributed.run --nproc_per_node=2 --rdzv_backend c10d --rdzv-endpoint=localhost:0 $train $hparams \
    --data_folder $data_folder  --grad_accumulation_factor 4 --output_folder $output_folder \
    --skip_prep true --lr $lr --log_interval 500 --number_of_epochs 10 --train_csv $train_csv


lr=0.0004
output_folder=results/all_lr_${lr}

python -m torch.distributed.run --nproc_per_node=2 --rdzv_backend c10d --rdzv-endpoint=localhost:0 $train $hparams \
    --data_folder $data_folder  --grad_accumulation_factor 4 --output_folder $output_folder \
    --skip_prep true --lr $lr --log_interval 500 --number_of_epochs 10 --train_csv $train_csv


lr=0.0016
output_folder=results/all_lr_${lr}

python -m torch.distributed.run --nproc_per_node=2 --rdzv_backend c10d --rdzv-endpoint=localhost:0 $train $hparams \
    --data_folder $data_folder  --grad_accumulation_factor 4 --output_folder $output_folder \
    --skip_prep true --lr $lr --log_interval 500 --number_of_epochs 10 --train_csv $train_csv
