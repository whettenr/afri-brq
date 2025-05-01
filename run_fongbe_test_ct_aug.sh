#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=i❤️🇧🇯
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=2
# SBATCH --nodelist=hemera
#SBATCH --constraint='GPURAM_Min_24GB&GPURAM_Max_32GB'
#SBATCH --mail-type=END,FAIL


conda activate aa
cd /users/rwhetten/african_brq

train=/users/rwhetten/african_brq/quantization_framework/train_new_targets_aug.py
hparams=/users/rwhetten/african_brq/quantization_framework/BEST-RQ_aug.yaml
data_folder=/users/fkponou/data/speechbrain/To_Ryan
train_csv=/users/rwhetten/african_brq/quantization_framework/targets/kmeans_train_tgts.csv
valid_csv=/users/rwhetten/african_brq/quantization_framework/targets/kmeans_valid_tgts.csv


lr=0.0016
output_folder=results/fongbe_lr_0.0016_ct_200e_aug
python -m torch.distributed.run --nproc_per_node=2 --rdzv_backend c10d --rdzv-endpoint=localhost:0 $train $hparams --find_unused_parameters \
    --data_folder $data_folder  --grad_accumulation_factor 8 --output_folder $output_folder \
    --skip_prep true --lr $lr --log_interval 500 --number_of_epochs 250 \
    --train_csv $train_csv --valid_csv $valid_csv \
    --num_workers 2 \
    --seconds_per_batch 300 \
    --reset_lin True

# lr=0.0016
# output_folder=results/fongbe_lr_0.0016_ct_200e
# python -m torch.distributed.run --nproc_per_node=2 --rdzv_backend c10d --rdzv-endpoint=localhost:0 $train $hparams --find_unused_parameters \
#     --data_folder $data_folder  --grad_accumulation_factor 8 --output_folder $output_folder \
#     --skip_prep true --lr $lr --log_interval 500 --number_of_epochs 250 \
#     --train_csv $train_csv --valid_csv $valid_csv \
#     --num_workers 2 \
#     --seconds_per_batch 300 \
#     --reset_lin False --lin_dim 500
