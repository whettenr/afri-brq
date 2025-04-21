import sys
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)

        # Forward pass
        feats = self.modules.hubert(wavs, wav_lens)
        x = self.modules.enc(feats)
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        return p_ctc, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC) given predictions and targets."""

        p_ctc, wav_lens = predictions

        ids = batch.id
        tokens, tokens_lens = batch.tokens
        # tokenizer = hparams["tokenizer"]
        # Label Augmentation
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            tokens = self.hparams.wav_augment.replicate_labels(tokens)
            tokens_lens = self.hparams.wav_augment.replicate_labels(tokens_lens)

        loss = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            
            # predicted_words = self.tokenizer(sequence, task="decode_from_list")
            predicted_words = [
                    tokenizer.sp.decode_ids(utt_seq).split(" ") for utt_seq in sequence
                ]
            # predictions = [
            #         tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
            #     ]
                
            # Convert indices to words
            # target_words = undo_padding(tokens, tokens_lens)
            # target_words = self.tokenizer(target_words, task="decode_from_list")
            
            target_words = [wrd.split(" ") for wrd in batch.utterance]
            
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_hubert, new_lr_hubert = self.hparams.lr_annealing_hubert(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            if not self.hparams.hubert.freeze:
                sb.nnet.schedulers.update_learning_rate(
                    self.hubert_optimizer, new_lr_hubert
                )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_hubert": old_lr_hubert,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]},
                min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the hubert optimizer and model optimizer"

        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        # If the wav2vec encoder is unfrozen, we create the optimizer
        if not self.hparams.hubert.freeze:
            self.hubert_optimizer = self.hparams.hubert_opt_class(
                self.modules.hubert.parameters()
            )
            if self.checkpointer is not None:
                self.checkpointer.add_recoverable(
                    "hubert_opt", self.hubert_optimizer
                )
            self.optimizers_dict = {
                "hubert_optimizer": self.hubert_optimizer,
                "model_optimizer": self.model_optimizer,
            }
        else:
            self.optimizers_dict = {"model_optimizer": self.model_optimizer}

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

    def freeze_optimizers(self, optimizers):
        """Freezes the hubert optimizer according to the warmup steps"""
        valid_optimizers = {}
        if not self.hparams.hubert.freeze:
            valid_optimizers["hubert_optimizer"] = optimizers[
                "hubert_optimizer"
            ]
        valid_optimizers["model_optimizer"] = optimizers["model_optimizer"]
        return valid_optimizers


# Define custom data procedure
def dataio_prepare(hparams,tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    # Define audio pipeline. In this case, we simply read the path contained
    # in the variable wav with the audio reader.
    @sb.utils.data_pipeline.takes("filename")
    @sb.utils.data_pipeline.provides("sig","duration")
    def audio_pipeline(wav):
        wav = f"/users/fkponou/data/for_asr/fongbe_one/{wav}.wav"
        sig = sb.dataio.dataio.read_audio(wav)
        sig = sig.squeeze(0)
        yield sig
        duration = len(sig)/hparams["sample_rate"]
        yield duration

    @sb.utils.data_pipeline.takes("filename")
    @sb.utils.data_pipeline.provides("sig","duration")
    def sp_audio_pipeline(wav):
        wav = f"/users/fkponou/data/for_asr/fongbe_one/{wav}.wav"
        # print(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        sig = sig.unsqueeze(0)
        sig = hparams["speed_perturb"](sig)
        sig = sig.squeeze(0)
        yield sig
        duration = len(sig)/hparams["sample_rate"]
        yield duration

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("message")
    @sb.utils.data_pipeline.provides(
        "utterance", "tokens_list", "tokens_bos", "tokens_eos","tokens"
    )
    def reference_text_pipeline(translation):
        """Processes the transcriptions to generate proper labels"""
        yield translation
        tokens_list = tokenizer.sp.encode_as_ids(translation)
        yield tokens_list
        tokens_bos =  torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    datasets = {}
    data_folder = hparams["data_folder"]
    for dataset in ["train", "valid"]:
        csv_path  = hparams[f"{dataset}_csv"]

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
                "tokens_eos",
                "tokens"
            ],
        )

    for dataset in ["test"]:
        csv_path  = hparams[f"{dataset}_csv"]
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
                "tokens_eos",
                "tokens"
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

    # return train_data, valid_data, test_data


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)
        
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["train_csv"],
        annotation_read="message",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
    )
    
    # Create the datasets objects as well as tokenization and encoding :-D
    datasets = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    
    # Adding objects to trainer.
    asr_brain.tokenizer = tokenizer

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Test
    asr_brain.evaluate(
        datasets["test"],
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )