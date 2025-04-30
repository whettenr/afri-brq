#!/bin/bash -l
#SBATCH --job-name=ASR
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=15:00:00
#SBATCH --mem=32G  
#SBATCH --constraint='GPURAM_Min_24GB&GPURAM_Max_32GB'


# conda activate aa
# cd /users/rwhetten/african_brq
# train=/users/rwhetten/african_brq/fine_tune/train_with_bestrq.py
# hparams=/users/rwhetten/african_brq/fine_tune/train_sb_BEST-RQ.yaml
# pt_model_path=/users/rwhetten/african_brq/results/fongbe_lr_0.0008/save/CKPT+2025-04-10+04-54-59+00
# output_folder=results/test_0.0008_epoch100
# python $train $hparams --pt_model_path $pt_model_path --output_folder $output_folder


# conda activate aa
# cd /users/rwhetten/african_brq
# train=/users/rwhetten/african_brq/fine_tune/train_with_bestrq.py
# hparams=/users/rwhetten/african_brq/fine_tune/train_sb_BEST-RQ.yaml
# output_folder=results/test_0.0016_epoch200_exclude_2Lin_noaug # change this
# pt_model_path=/users/rwhetten/african_brq/results/fongbe_lr_0.0016/CKPT+2025-04-11+14-37-42+00
# python $train $hparams --pt_model_path $pt_model_path --output_folder $output_folder


conda activate aa
cd /users/rwhetten/african_brq
train=/users/rwhetten/african_brq/fine_tune/train_with_bestrq.py
hparams=/users/rwhetten/african_brq/fine_tune/train_sb_BEST-RQ.yaml
output_folder=results/test_0.0016_epoch250_kmct
pt_model_path=/users/rwhetten/african_brq/results/fongbe_lr_0.0016_ct_200e/CKPT-250
python $train $hparams --pt_model_path $pt_model_path --output_folder $output_folder


conda activate aa
cd /users/rwhetten/african_brq
train=/users/rwhetten/african_brq/fine_tune/train_with_bestrq.py
hparams=/users/rwhetten/african_brq/fine_tune/train_sb_BEST-RQ.yaml
output_folder=results/test_0.0016_epoch250_kmct_ptaug
pt_model_path=/users/rwhetten/african_brq/results/fongbe_lr_0.0016_ct_200e_aug/CKPT-250
python $train $hparams --pt_model_path $pt_model_path --output_folder $output_folder


conda activate aa
cd /users/rwhetten/african_brq
train=/users/rwhetten/african_brq/fine_tune/train_with_bestrq.py
hparams=/users/rwhetten/african_brq/fine_tune/train_sb_BEST-RQ.yaml
output_folder=results/test_0.0016_epoch250_rqct
pt_model_path=/users/rwhetten/african_brq/results/fongbe_lr_0.0016_ct_200e_rq/CKPT-250
python $train $hparams --pt_model_path $pt_model_path --output_folder $output_folder





conda activate aa
cd /users/rwhetten/african_brq
train=/users/rwhetten/african_brq/fine_tune/train_with_bestrq.py
hparams=/users/rwhetten/african_brq/fine_tune/train_sb_BEST-RQ.yaml
output_folder=results/test_all_3_126epochs
pt_model_path=/users/rwhetten/african_brq/results/all_3_lr_0.0016/CKPT-126
python $train $hparams --pt_model_path $pt_model_path --output_folder $output_folder




conda activate aa
cd /users/rwhetten/african_brq
train=/users/rwhetten/african_brq/fine_tune/train_with_bestrq.py
hparams=/users/rwhetten/african_brq/fine_tune/train_sb_BEST-RQ.yaml
output_folder=results/test_0.0016_epoch250
pt_model_path=/users/rwhetten/african_brq/results/fongbe_lr_0.0016/save/CKPT+2025-04-26+13-04-54+00
python $train $hparams --pt_model_path $pt_model_path --output_folder $output_folder




conda activate aa
cd /users/rwhetten/african_brq
train=/users/rwhetten/african_brq/fine_tune/train_with_bestrq.py
hparams=/users/rwhetten/african_brq/fine_tune/train_sb_BEST-RQ.yaml
output_folder=results/test_0.0008_epoch200
pt_model_path=/users/rwhetten/african_brq/results/fongbe_lr_0.0008/save/CKPT+2025-04-11+14-07-52+00
python $train $hparams --pt_model_path $pt_model_path --output_folder $output_folder

