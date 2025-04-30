#!/usr/bin/env python3
import sys
import torch
from hyperpyyaml import load_hyperpyyaml
from sacremoses import MosesDetokenizer
from torch.nn.parallel import DistributedDataParallel
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


# Define training procedure
class ST(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig  
        tokens_bos, _ = batch.tokens_bos  

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
            tokens_bos = self.hparams.wav_augment.replicate_labels(tokens_bos)

        
        # run through brq 
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        feats = self.modules.brq[0](feats) # CNNs
        src = self.modules.brq[1](feats, wav_lens) # Conformer layers

        src = self.modules.enc(src)
        dec_out = self.modules.NLLB(
            src, tokens_bos, pad_idx=self.hparams.pad_index
        )
        # logits and softmax
        p_seq = self.hparams.log_softmax(dec_out)
        if hparams["nllb_frozen"] and not p_seq.requires_grad:
            p_seq.requires_grad = True

        # compute outputs
        hyps = None
        if stage == sb.Stage.VALID and self.optimizer_step >= 1000:
            # the output of the encoder (enc) is used for valid search
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                if isinstance(self.modules.NLLB, DistributedDataParallel):
                    self.modules.NLLB = self.modules.NLLB.module
                hyps, _, _, _ = self.hparams.valid_search(
                    src.detach(), wav_lens
                )

        elif stage == sb.Stage.TEST:
            if isinstance(self.modules.NLLB, DistributedDataParallel):
                self.modules.NLLB = self.modules.NLLB.module
            hyps, _, _, _ = self.hparams.valid_search(src.detach(), wav_lens)

        return p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""
        (p_seq, wav_lens, hyps) = predictions
        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        tokens_eos = self.modules.NLLB.custom_padding(
            tokens_eos,
            0,
            self.modules.NLLB.model.model.decoder.config.pad_token_id,
        )

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            tokens_eos = self.hparams.wav_augment.replicate_labels(tokens_eos)
            tokens_eos_lens = self.hparams.wav_augment.replicate_labels(
                tokens_eos_lens
            )

        loss = self.hparams.seq_cost(p_seq, tokens_eos, length=tokens_eos_lens)
        fr_detokenizer = MosesDetokenizer(lang=self.hparams.lang)

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                and self.optimizer_step >= 1000
                or (stage == sb.Stage.TEST)
            ):
                detokenized_translation = [
                    fr_detokenizer.detokenize(translation.split(" "))
                    for translation in batch.utterance
                ]
                targets = [detokenized_translation]
                
                predictions = [
                    fr_detokenizer.detokenize(hyp.split(" "))
                    for hyp in self.modules.NLLB.tokenizer.batch_decode(
                        hyps, skip_special_tokens=True
                    )
                ]
                
                self.bleu_metric.append(ids, predictions, targets)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)


        return loss

    def init_optimizers(self):
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters()
        )

        self.optimizers_dict = {"model_optimizer": self.adam_optimizer}

        # Initializes the hubert optimizer if the model is not hubert_frozen
        if not self.hparams.hubert_frozen:
            self.hubert_optimizer = self.hparams.hubert_opt_class(
                self.modules.brq.parameters()
            )
            self.optimizers_dict["hubert_optimizer"] = self.hubert_optimizer

        if not self.hparams.nllb_frozen:
            self.nllb_optimizer = self.hparams.nllb_opt_class(
                self.modules.NLLB.parameters()
            )
            self.optimizers_dict["nllb_optimizer"] = self.nllb_optimizer

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            if not self.hparams.hubert_frozen:
                self.hparams.lr_annealing_hubert(
                    self.hubert_optimizer, self.optimizer_step
                )
            if not self.hparams.nllb_frozen:
                self.hparams.lr_annealing_nllb(
                    self.nllb_optimizer, self.optimizer_step
                )

    def on_stage_start(self, stage, epoch):
        """Gets called when a stage (either training, validation, test) starts."""
        self.bleu_metric = self.hparams.bleu_computer()

        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.bleu_metric = self.hparams.bleu_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_loss

        else:  # valid or test
            stage_stats = {"loss": stage_loss}
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                and self.optimizer_step >= 1000
                or stage == sb.Stage.TEST
            ):
                stage_stats["BLEU"] = self.bleu_metric.summarize(field="BLEU")
                stage_stats["BLEU_extensive"] = self.bleu_metric.summarize()
                self.anneal_bleu = stage_stats["BLEU"]

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():
            current_epoch = self.hparams.epoch_counter.current
            old_lr_adam, new_lr_adam = self.hparams.lr_annealing_adam(
                self.anneal_bleu  # stage_stats["BLEU"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.adam_optimizer, new_lr_adam
            )

            stats_meta = {
                "epoch": current_epoch,
                "steps": self.optimizer_step,
                "lr_adam": old_lr_adam,
            }

            if not self.hparams.hubert_frozen:
                self.hparams.lr_annealing_hubert(
                    self.hubert_optimizer, self.optimizer_step
                )
                stats_meta["lr_hubert"] = self.hubert_optimizer.param_groups[
                    0
                ]["lr"]
            if not self.hparams.nllb_frozen:
                self.hparams.lr_annealing_nllb(
                    self.nllb_optimizer, self.optimizer_step
                )
                stats_meta["lr_nllb"] = self.nllb_optimizer.param_groups[0][
                    "lr"
                ]
            self.hparams.train_logger.log_stats(
                stats_meta=stats_meta,
                train_stats={"loss": self.train_stats},
                valid_stats=stage_stats,
            )

            # create checkpoint
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                and self.optimizer_step >= 1000
            ):
                meta = {"BLEU": stage_stats["BLEU"], "epoch": current_epoch}
                name = "checkpoint_epoch" + str(current_epoch)

                self.checkpointer.save_and_keep_only(
                    meta=meta, name=name, num_to_keep=10, max_keys=["BLEU"]
                )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

            with open(self.hparams.bleu_file, "w") as w:
                self.bleu_metric.write_stats(w)


# Define custom data procedure
def dataio_prepare(hparams,tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    # Define audio pipeline. In this case, we simply read the path contained
    # in the variable wav with the audio reader.
    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("sig","duration")
    def audio_pipeline(wav):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        sig = sig.squeeze(0)
        yield sig
        duration = len(sig)/hparams["sample_rate"]
        yield duration

    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("sig","duration")
    def sp_audio_pipeline(wav):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        sig = sig.unsqueeze(0)
        sig = hparams["speed_perturb"](sig)
        sig = sig.squeeze(0)
        yield sig
        duration = len(sig)/hparams["sample_rate"]
        yield duration

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("utterance")
    @sb.utils.data_pipeline.provides(
        "utterance", "tokens_list", "tokens_bos", "tokens_eos"
    )
    def reference_text_pipeline(translation):
        """Processes the transcriptions to generate proper labels"""
        yield translation
        labels = tokenizer(
            text_target=translation.replace("\n", ""), return_tensors="pt"
        )
        tokens_list = labels["input_ids"].tolist()[-1]
        yield tokens_list
        tokens_bos = torch.LongTensor(tokens_list[0:-1])
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list[1:])
        yield tokens_eos

    datasets = {}
    data_folder = hparams["data_folder"]
    for dataset in ["train", "valid"]:
        csv_path  = hparams[f"{dataset}_set"]

        is_use_sp = dataset == "train" and "speed_perturb" in hparams
        audio_pipeline_func = sp_audio_pipeline if is_use_sp else audio_pipeline

        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path =csv_path,
            replacements={"data_root": data_folder},
            dynamic_items=[audio_pipeline_func, reference_text_pipeline],
            output_keys=[
                "id",
                "sig",
                "duration",
                "utterance",
                "tokens_list",
                "tokens_bos",
                "tokens_eos"
            ],
        )

    for dataset in ["test"]:
        csv_path  = hparams[f"{dataset}_set"]
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path =csv_path,
            replacements={"data_root": data_folder},
            dynamic_items=[audio_pipeline, reference_text_pipeline],
            output_keys=[
                "id",
                "sig",
                "duration",
                "utterance",
                "tokens_list",
                "tokens_bos",
                "tokens_eos"
            ],
        )

    # Sorting training data with ascending order makes the code  much
    # faster  because we minimize zero-padding. In most of the cases, this
    # does not harm the performance.
    if hparams["sorting"] == "ascending":
        # use smaller dataset to debug the model
        if hparams["debug"]:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_max_value={"duration": hparams["avoid_if_longer_than"]},
                sort_key="duration",
                reverse=True,
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                key_max_value={"duration": hparams["avoid_if_longer_than"]},
                sort_key="duration",
                reverse=True,
            )
        else:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_max_value={"duration": hparams["avoid_if_longer_than"]},
                sort_key="duration",
                reverse=True,
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                key_max_value={"duration": hparams["avoid_if_longer_than"]},
                sort_key="duration",
                reverse=True,
            )

        hparams["dataloader_options"]["shuffle"] = False
        
    elif hparams["sorting"] == "descending":
        # use smaller dataset to debug the model
        if hparams["debug"]:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_max_value={"duration": hparams["avoid_if_longer_than"]},
                sort_key="duration",
                reverse=True,
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                key_max_value={"duration": hparams["avoid_if_longer_than"]},
                sort_key="duration",
                reverse=True,
            )
        else:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_max_value={"duration": hparams["avoid_if_longer_than"]},
                sort_key="duration",
                reverse=True,
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                key_max_value={"duration": hparams["avoid_if_longer_than"]},
                sort_key="duration",
                reverse=True,
            )

        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        # use smaller dataset to debug the model
        if hparams["debug"]:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_max_value={"duration": hparams["avoid_if_longer_than"]},
                sort_key="duration",
                reverse=True,
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                key_max_value={"duration": hparams["avoid_if_longer_than"]},
                sort_key="duration",
                reverse=True,
            )

        hparams["dataloader_options"]["shuffle"] = True
    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
  
    return datasets


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Create main experiment class
    st_brain = ST(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    st_brain.anneal_bleu = 0


    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams, st_brain.modules.NLLB.tokenizer)

    if "pretrainer" in hparams.keys() and hparams["brq_path"] is not None:
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected()


    # Training
    st_brain.fit(
        st_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Test
    for dataset in ["valid","test"]:
        st_brain.hparams.wer_file = (
            hparams["output_folder"] + "/wer_test" + ".txt"
        )
        st_brain.evaluate(
            datasets[dataset],
            test_loader_kwargs=hparams["test_dataloader_options"],
        )
