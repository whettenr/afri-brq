#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --job-name=i❤️🇧🇯
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
# SBATCH --nodelist=hemera
#SBATCH --constraint='GPURAM_Min_32GB'
#SBATCH --mail-type=END,FAIL


conda activate aa
cd /users/rwhetten/african_brq

train=train.py
hparams=/users/rwhetten/african_brq/BEST-RQ_600M.yaml
data_folder=/users/fkponou/data/speechbrain/To_Ryan
output_folder=/users/rwhetten/african_brq/results/continual_pretrainin_fongbe

python -m torch.distributed.run --nproc_per_node=1 --rdzv_backend c10d --rdzv-endpoint=localhost:0 $train $hparams --find_unused_parameters \
    --data_folder $data_folder  --grad_accumulation_factor 32 --output_folder $output_folder \
    --log_interval 500 --number_of_epochs 200