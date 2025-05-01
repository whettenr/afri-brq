#!/bin/bash

source_path="/users/rwhetten/african_brq"
target_path="uaj64gk@jean-zay.idris.fr:/gpfswork/rech/nkp/uaj64gk/afri_brq"

rsync -azh "$source_path" "$target_path" \
    --progress \
    --delete --force \
    --exclude="__pycache__/" \
    --exclude="results/" \
    --exclude="log/" \
    --exclude=".idea/" \
    --exclude="quantization_framework/embeddings" \
    --exclude="quantization_framework/targets" \
    --exclude="slurm" \
    --exclude="slurm*" \
    --exclude="dataaug/MUSAN" \
    --exclude="dataaug/rir"
    
    # --filter=":- .gitignore" \