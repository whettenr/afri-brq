#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --job-name=yor
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=2
# SBATCH --nodelist=hemera
#SBATCH --constraint='GPURAM_Min_24GB&GPURAM_Max_32GB'
#SBATCH --mail-type=END,FAIL

conda activate aa
cd /users/rwhetten/african_brq

run=quantization_framework/quantize_fbank_GPU.py
hparams=quantization_framework/BEST-RQ_fbanks.yaml


save_targets_folder=/users/rwhetten/african_brq/store
save_train_folder=train
save_valid_folder=valid




train_csv=csvs/yor.csv
valid_csv=csvs/valid.csv

python $run $hparams \
    --train_csv $train_csv \
    --valid_csv $valid_csv \
    --save_targets_folder $save_targets_folder \
    --save_train_folder $save_train_folder \
    --save_valid_folder $save_valid_folder \
    --seconds_per_batch 300 \
    --train_num_buckets 70 \
    --skip_train false \
    --skip_valid true
