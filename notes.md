# Training BRQ on african languages

## April 8th, 2025
- create repo
    - add yaml file and train.py
    - ajust train file
    - prepare csv
- thoughts
    - add augmentation in pretraining?
    - add denoising type augmentation? WavLM style?

```bash
# command for find and replace in terminal
# for librispeech
sed -i 's/\/corpus/\/lustre\/fsmisc\/dataset/g' csvs_jz/new/*.csv 
# for other languages
sed -i 's/users\/fkponou\/data\/speechbrain\/To_Ryan/lustre\/fsn1\/projects\/rech\/nkp\/uaj64gk\/african_brq/g' csvs_jz/new/*.csv

ln -s /lustre/fsmisc/dataset/MUSAN /lustre/fswork/projects/rech/nkp/uaj64gk/afri_brq/african_brq/dataaug/musan
ln -s /lustre/fsmisc/dataset/MUSAN /users/rwhetten/african_brq/dataaug/musan

# test on 1 a100
srun -A nkp@a100 -C a100 --gres=gpu:1 --time=01:00:00 --cpus-per-task=16 --nodes=1 --pty /bin/bash
srun -c 8 -p gpu --gpus-per-node=2  --constraint='GPURAM_Min_24GB&GPURAM_Max_24GB' --mem=32G --time=04:00:00 
--pty /bin/bash
```

todo
- generate FBank targets offline
- get augmentation set up 
    - in hparams
    - csvs
    - in train