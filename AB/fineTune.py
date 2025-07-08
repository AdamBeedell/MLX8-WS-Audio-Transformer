### whisper small fine-tune
## currently for a better single word handling


## import stuff

import torch  # base ML library

import transformers # huggingface transformers library
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer # Openai whisper stuff ## dont really need the last 2 except for test block
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer # generic sequence to sequence stuff - Here the audio counts as a sequence as does the transcription as they're both in time. usually images wouldnt be appropriate, but our image conversion preserves time.
import datasets
from datasets import Audio, dataset_dict # HF datasets library
import evaluate # HF evaluation library for loss functions and metrics
import huggingface_hub # HF logins and training functions

import wandb # weights and biases for saving and logging modols and metrics\

import os # refer to local files
import pathlib # also refer to local files, may not need

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import accelerate



## Testing block

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="en", task="transcribe")

input_str = "Asmorandamardicadaistinaculdacar" #### word to train on
AsmoTokens = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(AsmoTokens, skip_special_tokens=False)
decoded_str = tokenizer.decode(AsmoTokens, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")



##### Main script

#### ref https://huggingface.co/blog/fine-tune-whisper

### Create dataset

## Rough Order of Operations::
## Find data in Data/Memos
## Build list of dicts (dataset dict is a specific thing, but i've done a generic thing, i dont know what the off-the-shelf one does)
## list of dicts -> Dataset
## Dataset -> dataset with correct audio sampling rate
## Dataset -> dataset with padding and tokenized text and spectogram log_mel
## Dataset -> dataset with batches

# Note: Because I'm following a huggingface example, the dataloader step that would usually be handling the dataset -> batches is handled by this huggingface library, they have Collators that forfil basically the same role, but pass to a dataloader under the hood.

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="en", task="transcribe")  ### This will do audio -> log_mel chart and text to token IDs, also handles padding and special tokens

### build dataset list

data = []

for item in pathlib.Path("Data/Memos").glob("*.wav"):
    data.append({
        "audio": str(item), # string path to the audio file
        "sentence": input_str # In the first iteration we're just training one word, this needs to be varied later for Phthisis, Fblthp and Borborgrygmos
    })


## list to dataset

#data = list(data.values())  # just grab the values
dataset = datasets.Dataset.from_list(data)

## dataset to dataset with proper audio column
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

## dataset to dataset with padding and tokenized text and spectogram log_mel

def prepare_dataset(batch):   #### This is a bit misleading, should occur across the whole dataset and then batching gets handled by the collator, this is a HF format, may remove for clarity
    
    audio = batch["audio"] # load and resample audio data from ??(but should be 16kHz) to 16kHz
    result = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["sentence"]) #### This is the thing that does the audio -> log_mel spectogram conversion
    batch["input_features"] = result["input_features"][0]
    batch["labels"] = result["labels"]

    return batch


dataset = dataset.map(prepare_dataset, remove_columns=["audio", "sentence"])  ## Does the function above, makes 2 new NN readable columns, drop the human ones

## dataset to dataset with batches

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # separate inputs and labels
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
)

train_dataset=dataset
eval_dataset=dataset

### load model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")  ## Small runs OK
model.generation_config.language = "en"  ## English
model.generation_config.task = "transcribe" ## Transcription mode
model.generation_config.forced_decoder_ids = None




### validation

## Loss/Eval function

metric = evaluate.load("wer")  ### prebuilt loss function helper for Word Error Rate - Includes swapping, insertion and deletion errors in the calculation - See Notes MLX day 22

def compute_metrics(pred):  
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}



training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-hi",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=1,
    max_steps=50,
    gradient_checkpointing=True,
    fp16=False,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=50,
    eval_steps=10,
    logging_steps=10,
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.tokenizer,
)



## Do training
trainer.train()  # This will train the model, save it to the output directory and push it to the hub if configured
trainer.save_model()  # Save the model to the output directory

