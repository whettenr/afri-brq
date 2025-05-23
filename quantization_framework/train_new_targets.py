#!/usr/bin/env python3
"""Recipe for pretraining Best-RQ (https://arxiv.org/pdf/2405.04296)

To run this recipe call python train.py BEST-RQ.yaml --find_unused_parameters

Authors
    * Ryan Whetten 2025
"""

import sys
import time
from functools import partial

import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.lobes.models.BESTRQ import brq_mask_collate_tgts_fn as brq_mask_collate_fn
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger
import numpy as np

logger = get_logger(__name__)


class BestRQBrain(sb.core.Brain):

    def compute_forward(self, batch, stage):
        """Computes forward pass through BestRQ model and returns encoded and
        target embeddings as well as other metrics of interest.
        """
        # get batch and mask
        wavs, wav_lens, mask, tgts = batch
        wavs, wav_lens, mask, tgts = (
            wavs.to(self.device),
            wav_lens.to(self.device),
            mask.to(self.device),
            tgts.to(self.device),
        )

        ### get fbanks and normalize
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        divis_by = self.hparams.pad_to_divisible_by
        feats = pad_feats(feats, divis_by)

        # get masked targets
        mask_idx = mask[::4] // 4
        B, T, C = feats.shape
        tgts = tgts[:, mask_idx]

        # generate random noise
        noise = torch.normal(
            mean=self.hparams.noise_mean,
            std=self.hparams.noise_std,
            size=(B, mask.shape[0], C),
            device=self.device,
        )
        # replace with random noise
        feats[:, mask, :] = noise

        #### convolutions
        src = self.modules.CNN(feats)

        ##### transformer
        enc_out = self.modules.wrapper(src, wav_lens)  # only use encoder

        ##### linear
        logits = self.modules.linear(enc_out)

        ##### get masked region for loss computation only over these.
        logits = logits[:, mask_idx, :]

        B, T, C = logits.shape
        return logits.view(B * T, C), tgts.view(B * T)

    def compute_objectives(self, predictions, batch, stage):
        pred, targets = predictions

        if stage != sb.Stage.TRAIN and sb.utils.distributed.if_main_process():
            predicted_classes = torch.argmax(pred, dim=-1)
            correct_predictions = predicted_classes == targets
            accuracy = correct_predictions.sum().item() / len(
                correct_predictions
            )
            self.acc_metric.append(accuracy)

        return F.cross_entropy(pred, targets)

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Called after fit_batch(), updates learning rate and does per-step logging."""

        if should_step:
            self.hparams.noam_annealing(self.optimizer)

        # Perform step-wise logging
        if (
            hasattr(self.hparams, "log_interval")
            and self.optimizer_step % self.hparams.log_interval == 0
        ):

            # Create a dictionary and fill it with everything we
            # want to log such as contrastive loss, diversity loss,
            # learning rate etc.
            log_dct = {}

            current_lr = self.optimizer.param_groups[0]["lr"]
            log_dct["steps"] = self.optimizer_step
            log_dct["lr"] = current_lr
            log_dct["avg_loss"] = self.avg_train_loss

            if hasattr(self, "time_last_log"):
                run_time_since_last_log = time.time() - self.time_last_log
                log_dct["run_time"] = run_time_since_last_log
            self.time_last_log = time.time()

            if sb.utils.distributed.if_main_process():
                self.hparams.train_steps_logger.log_stats(
                    stats_meta=log_dct,
                )


    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = []

    def on_stage_end(self, stage, stage_loss, epoch=None):

        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        if stage == sb.Stage.VALID:
            if self.acc_metric:

                stage_stats["accuracy"] = sum(self.acc_metric) / len(
                    self.acc_metric
                )

            self.hparams.train_stage_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "steps": self.optimizer_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            self.checkpointer.save_and_keep_only(
                end_of_epoch=True,
                num_to_keep=5,
                meta={"valid_loss": stage_loss},
            )
    
    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp.

        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """
        # Run this *after* starting all processes since jit/compiled modules
        # cannot be pickled.
        self._compile()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # Load latest checkpoint to resume training if interrupted
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible()
        
        if self.hparams.reset_lin:
            print(f"RESETTING LINEAR TO: {self.hparams.reset_lin_dim}")
            self.modules.linear = self.hparams.new_linear.to(device=self.device)
            self.init_optimizers()

            if self.checkpointer is not None:
                    self.checkpointer.add_recoverable("linear", self.modules.linear)

        # # try to load model
        # try: 
        #     # Initialize optimizers after parameters are configured
        #     self.init_optimizers()

        #     # Load latest checkpoint to resume training if interrupted
        #     if self.checkpointer is not None:
        #         self.checkpointer.recover_if_possible()
        #     for param in self.modules.wrapper.module.transformer.encoder.layers[0].parameters():
        #         print(f"Layer 0 grad : {param}")
        #         break

        # except:                
        #     # reinitialize optimizers
        #     self.init_optimizers()
        #     for param in self.modules.wrapper.module.transformer.encoder.layers[0].parameters():
        #         print(f"Layer 0 grad : {param}")
        #         break


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

    @sb.utils.data_pipeline.takes("wav", "start", "stop", "tgts")
    @sb.utils.data_pipeline.provides("sig", "targets")
    def audio_pipeline(wav,  start, stop, tgts):
        sig = sb.dataio.dataio.read_audio({
            "file": wav,
            "start": int(start),
            "stop": int(stop),
        })
        yield sig
        targets = torch.from_numpy(np.load(tgts)).long()
        yield targets


    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "targets"])

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
        brq_mask_collate_fn,
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


def main():
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    hparams.update(run_opts)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )


    # Part that matters starts here.
    train_dataset, valid_loader, train_loader_kwargs = dataio_prepare(hparams)

    brain = BestRQBrain(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )


    brain.fit(
        brain.hparams.epoch_counter,
        train_dataset,
        valid_loader,
        train_loader_kwargs=train_loader_kwargs,
        progressbar=True,
    )


if __name__ == "__main__":
    main()
