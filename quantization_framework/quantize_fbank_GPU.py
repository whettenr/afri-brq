import os
import sys
import pickle
import logging
from functools import partial

import torch
import torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.lobes.models.BESTRQ import brq_mask_collate_ids_fn
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

import pandas as pd
import numpy as np
from tqdm import tqdm

# print logs to stdout
logger = get_logger(__name__)

# BestRQ class (only with forward function)
class BestRQBrain(sb.core.Brain):

    def compute_forward(self, batch):
        """Computes forward pass through BestRQ model and returns encoded and
        target embeddings as well as other metrics of interest.
        """
        # get batch and mask
        ids, wavs, wav_lens, mask = batch
        wavs, wav_lens = (
            wavs.to(self.device),
            wav_lens.to(self.device),
        )

        ### get fbanks, normalize, pad
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)
        divis_by = self.hparams.pad_to_divisible_by
        feats = pad_feats(feats, divis_by)

        B, T, C = feats.shape
        targets = self.modules.Quantizer(
            feats.view(B, feats.shape[1] // divis_by, -1)
        )
        return targets


def pad_feats(feats, divis_by):
    """BEST-RQ quantizer stackes frames together. Hence, we need to pad the
    incoming features such that the time dimension is divisible by divis_by.

    Arguments
    ---------
    feats: torch.Tensor
        The feature tensor.
    divis_by: int
        The stacking factor. The time dimension of feats will become divisible
        by this value.

    Returns
    -------
    Padded features
    """

    B, T, C = feats.shape

    #### pad features to enable a reduction by pad_to_divisible_by for the
    # quantiser of BEST-RQ
    current_dim_size = T
    dim_to_pad = 1  # Pad along the second dimension (i.e. time)

    # Calculate the amount of padding needed to make the tensor divisible
    # by divis_by
    current_dim_size = feats.shape[dim_to_pad]
    # Ensure positive padding
    padding_needed = (divis_by - (current_dim_size % divis_by)) % divis_by

    # Define the padding
    # Initialize padding for all dimensions, have a look at the documentation of
    # torch.nn.functional.pad because the padding argument is quite special.
    padding = [0, 0, 0, 0, 0, 0]
    padding[dim_to_pad * 2] = (
        padding_needed  # Set padding for the chosen dimension
    )

    # add in padding to features and mask
    return torch.nn.functional.pad(feats, padding)

# load hparams
hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
with open(hparams_file, encoding="utf-8") as fin:
    hparams = load_hyperpyyaml(fin, overrides)

# initialize brain class
print("init brain")
brain = BestRQBrain(
    modules=hparams["modules"],
    opt_class=hparams["optimizer"],
    hparams=hparams,
    run_opts=run_opts,
    checkpointer=hparams["checkpointer"],  
)

print('device: ', brain.device)
print(f"output hidden {hparams['output_hidden_states']}")



def dataio_prepare(hparams):

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
    )

    # We remove longer and shorter files from the train.
    train_data = train_data.filtered_sorted(
        sort_key="duration",
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
        key_min_value={"duration": hparams["avoid_if_shorter_than"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
    )

    datasets = [train_data, valid_data]

    def get_output_lengths(input_lengths):
        """Function to get the output length of the feature extractor this is
        necessary to compute the masks of BestRQ.
        """
        sr = hparams["sample_rate"]
        hop_length = hparams["hop_length"]

        return (input_lengths // (sr * hop_length / 1000) + 1).to(torch.long)


    @sb.utils.data_pipeline.takes("wav", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop):
        sig = sb.dataio.dataio.read_audio({
            "file": wav,
            "start": int(start),
            "stop": int(stop),
        })
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig"])

    # We create the DynamicBatch Sampler
    train_sampler = DynamicBatchSampler(
        train_data,
        hparams["seconds_per_batch"],
        num_buckets=hparams["train_num_buckets"],
        length_func=lambda x: x["duration"],
        batch_ordering="random",
        shuffle=True,
    )

    # We define the custom collation function that is necessary for best-rq to
    # generate masks.
    brq_mask_collate_fn_partial = partial(
        brq_mask_collate_ids_fn,
        get_out_len_fn=get_output_lengths,
        mask_prob=hparams["mask_prob"],
        mask_length=hparams["mask_length"],
        n_mels=hparams["n_mels"],
    )

    train_loader_kwargs = {
        "batch_sampler": train_sampler,
        "collate_fn": brq_mask_collate_fn_partial,
        "num_workers": hparams["train_dataloader_options"]["num_workers"],
        "pin_memory": True,
    }

    valid_loader = SaveableDataLoader(
        valid_data,
        collate_fn=brq_mask_collate_fn_partial,
        num_workers=hparams["test_dataloader_options"]["num_workers"],
        batch_size=hparams["test_dataloader_options"]["batch_size"],
        pin_memory=True,
    )

    return train_data, valid_loader, train_loader_kwargs

# prepare data_loaders
train_dataset, valid_loader, train_loader_kwargs = dataio_prepare(hparams)
train_loader = brain.make_dataloader(train_dataset, stage=sb.Stage.TRAIN, **train_loader_kwargs)

# set model into eval mode (i.e. non-training/determanistic mode)
brain.modules.eval()

def save_tensor_slices_npy(tensor, ids, output_dir):
    # Move tensor to CPU and convert to numpy
    tensor = tensor.cpu().numpy()
    # .astype(np.uint16)
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    for i, id_ in enumerate(ids):
        # Get the i-th slice
        slice_array = tensor[i]
        # Define filename
        filename = os.path.join(output_dir, f"{id_}.npy")
        # Save using numpy
        directory = os.path.dirname(filename)
        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        np.save(filename, slice_array.astype(np.uint16))


# creating folder to save quantized targets for each ds if it doesn't already exits
save_targets_folder = hparams["save_targets_folder"]
save_train_folder = hparams["save_train_folder"]
save_valid_folder = hparams["save_valid_folder"]


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")

train_save_location = f"{save_targets_folder}/{save_train_folder}/"
valid_save_location = f"{save_targets_folder}/{save_valid_folder}/"
create_dir(save_targets_folder)
create_dir(train_save_location)
create_dir(valid_save_location)


# load quantizer
quantizer_params_location = f"{save_targets_folder}/FBankQuantizer.pth"
if os.path.exists(quantizer_params_location):
    print(f"Loading layer quantizer from {quantizer_params_location}")
    brain.modules.Quantizer.load_state_dict(torch.load(quantizer_params_location)) 
else:
    print(f"Layer quantizer not found in {quantizer_params_location}")

with torch.inference_mode():
    # go through train data_set

    print('in train')
    if not hparams['skip_train']:
        for b in tqdm(train_loader):
            tgts = brain.compute_forward(b)
            save_tensor_slices_npy(tgts, b[0], train_save_location)
    else:
        print('skipping train')

    # go through train data_set
    print('in valid')
    if not hparams['skip_valid']:
        for b in tqdm(valid_loader):
            tgts = brain.compute_forward(b)
            save_tensor_slices_npy(tgts, b[0], valid_save_location)
    else:
        print('skipping valid')



train_df = pd.read_csv(hparams['train_csv'])
train_df['tgts'] = train_save_location + train_df['ID'] + '.npy'
# train_df = train_df.drop(columns=['spk_id','wrd'])
csv_base_name = os.path.basename(hparams['train_csv'])
train_df.to_csv(f"{save_targets_folder}/tgts_{csv_base_name}", index=False)
print(f'duration train: {train_df.duration.sum()/ 3600} hours')

valid_csv_base_name = os.path.basename(hparams['valid_csv'])
if not hparams['skip_valid']:
    valid_df = pd.read_csv(hparams['valid_csv'])
    print(valid_save_location)
    valid_df['tgts'] = valid_save_location + valid_df['ID'].astype(str) + '.npy'
    # valid_df = valid_df.drop(columns=['spk_id','wrd'])
    valid_df.to_csv(f"{save_targets_folder}/tgts_{valid_csv_base_name}", index=False)
    print(f'duration valid: {valid_df.duration.sum() / 3600} hours')
else:
    print('skipping saving valid')


# save quantizer
if os.path.exists(quantizer_params_location):
    print(f"Layer quantizer alread exists, not saving {quantizer_params_location}")
else:
    torch.save(brain.modules.Quantizer.state_dict(), quantizer_params_location)
    print(f"Saving layer quantizer in {quantizer_params_location}")

exit(0)
