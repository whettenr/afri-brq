#!/bin/bash 


module load pytorch-gpu/py3/2.6.0
conda activate ft-sb
cd /lustre/fswork/projects/rech/nkp/uaj64gk/afri_brq/african_brq

run=quantization_framework/quantize_fbank_GPU.py
hparams=quantization_framework/BEST-RQ_fbanks.yaml

train_csv=csvs_jz/efiy.csv
valid_csv=csvs_jz/valid.csv

save_targets_folder=/lustre/fsn1/projects/rech/nkp/uaj64gk/african_brq/targets
save_train_folder=train
save_valid_folder=valid

python $run $hparams \
    --train_csv $train_csv \
    --valid_csv $valid_csv \
    --save_targets_folder $save_targets_folder \
    --save_train_folder $save_train_folder \
    --save_valid_folder $save_valid_folder \
    --seconds_per_batch 300 \
    --train_num_buckets 70 \
    --cb_vocab 2048 \
    --skip_valid false



train_csv=csvs_jz/cappfm.csv
valid_csv=csvs_jz/valid.csv

python $run $hparams \
    --train_csv $train_csv \
    --valid_csv $valid_csv \
    --save_targets_folder $save_targets_folder \
    --save_train_folder $save_train_folder \
    --save_valid_folder $save_valid_folder \
    --seconds_per_batch 300 \
    --train_num_buckets 70 \
    --cb_vocab 2048 \
    --skip_train false \
    --skip_valid false 


train_csv=csvs_jz/fiy.csv
valid_csv=csvs_jz/valid.csv

python $run $hparams \
    --train_csv $train_csv \
    --valid_csv $valid_csv \
    --save_targets_folder $save_targets_folder \
    --save_train_folder $save_train_folder \
    --save_valid_folder $save_valid_folder \
    --seconds_per_batch 300 \
    --train_num_buckets 70 \
    --cb_vocab 2048 \
    --skip_train false \
    --skip_valid true


train_csv=csvs_jz/librispeech/train.csv
valid_csv=csvs_jz/librispeech/dev-clean.csv

python $run $hparams \
    --train_csv $train_csv \
    --valid_csv $valid_csv \
    --save_targets_folder $save_targets_folder \
    --save_train_folder $save_train_folder \
    --save_valid_folder $save_valid_folder \
    --seconds_per_batch 300 \
    --train_num_buckets 70 \
    --cb_vocab 2048 \
    --skip_train false \
    --skip_valid false 