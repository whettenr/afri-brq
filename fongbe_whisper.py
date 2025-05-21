from datasets import load_dataset,Dataset,DatasetDict, Audio
import pandas as pd 
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from transformers import WhisperProcessor
import evaluate
import torch
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperForConditionalGeneration
from functools import partial
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer


from dataclasses import dataclass
from typing import Any, Dict, List, Union


metric = evaluate.load("wer")


train = pd.read_csv("/users/fkponou/data/for_asr/mymy_asr/train_sub.csv")
test = pd.read_csv("/users/fkponou/data/for_asr/mymy_asr/test_sub.csv")
valid = pd.read_csv("/users/fkponou/data/for_asr/mymy_asr/valid_sub.csv")

train_set = Dataset.from_pandas(train) 
test_set = Dataset.from_pandas(test) 
valid_set = Dataset.from_pandas(valid)

voice = DatasetDict({
    "train":train_set,
    "test":test_set,
    "valid":valid_set
})

voice = voice.select_columns(["file_path", "message"])



processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="yoruba", task="transcribe"
)

sampling_rate = processor.feature_extractor.sampling_rate

common_voice = voice.cast_column("file_path", Audio(sampling_rate=sampling_rate))

def prepare_dataset(example):
    audio = example["file_path"]

    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["message"],
    )

    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example

common_voice = common_voice.map(
    prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1
)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


normalizer = BasicTextNormalizer()


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]

    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}


model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

model.config.use_cache = False

model.generate = partial(
    model.generate, language="yoruba", task="transcribe", use_cache=True
)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
)


training_args = Seq2SeqTrainingArguments(
    output_dir="/users/fkponou/data/speechbrain/Whisper/whisper-small-fongbe",  
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    max_steps=500,  
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    eval_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)


if __name__ == "__main__" :
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )
