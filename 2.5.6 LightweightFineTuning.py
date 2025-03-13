# pip install -r 2.5.6\ requirements.txt
"""
Author: Wille, Krisztina Martina
Creation Time: 2025.03.12

GenAI Course 2, 5.6 LightweightFineTuning

Purpose: This script demonstrates the process of loading, training, and evaluating a GPT-2 model for sequence classification using parameter-efficient fine-tuning techniques such as LoRA (Low-Rank Adaptation) and PEFT (Parameter-Efficient Fine-Tuning). The script includes functions for loading the foundation model, training the GPT-2 model, performing LoRA fine-tuning, and performing PEFT fine-tuning. It also includes utility functions for tokenization, metric computation, and time conversion.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel, AutoPeftModelForSequenceClassification, TaskType
import numpy as np
import torch
import os
import shutil
import sys
import time
import traceback

def secToHuman(elapsed_time):
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    # return hours, minutes, seconds
    return f"{int(hours):02}:{int(minutes):02}:{seconds:.2f}"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Ensure it's not a dictionary
        logits = logits[0]
    predictions = np.argmax(logits, axis=1)
    return {"accuracy": (predictions == labels).mean()}


def tokenize_fn(examples):
    global tokenizer
    return tokenizer(examples["sms"], padding="max_length", truncation=True, max_length=256)

# variables
CHECKPOINTS = "./checkpoints"
GPT2_FINETUNED_MODEL = "./gpt2_tuned_model"
LORA_FINETUNED_MODEL = "./lora_tuned_model"
PEFT_FINETUNED_MODEL = "./peft_tuned_model"

label_names = ["not spam", "spam"]
id2label = {idx: label for idx, label in enumerate(label_names)}
label2id = {label: idx for idx, label in enumerate(label_names)}
model = None
peft_model = None
tokenizer = None
data_collator = None
tokenized_ds = {} 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###############################################################################
# Loading and Evaluating a Foundation Model
###############################################################################

def load_foundation_model():
    global model, tokenizer, data_collator, label_names, id2label, label2id, device
    split = ['train', 'test']

    raw_dataset = load_dataset("sms_spam")
    full_dataset = raw_dataset['train'].train_test_split(test_size=0.2, seed=42, shuffle=True)
    
    dataset =  {"train": full_dataset["train"].shuffle(seed=42).select(range(1000)),  # Keep only 1000 samples
                "test":  full_dataset["test"].shuffle(seed=42).select(range(200))  # Keep only 200 samples
                }

    # Load GPT-2 tokenizer and model
    model_name = "gpt2"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label=id2label, label2id=label2id
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT-2 doesn't have a padding token, so use eos_token and set padding_side to left
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Ensure left padding for GPT-2

    for s in split:
        tokenized_ds[s] = dataset[s].map(tokenize_fn, batched=True)

    tokenized_ds["train"] = tokenized_ds["train"].map(
        lambda e: {'labels': e['label']},  
        batched=True,
        remove_columns=['label']
    )
    tokenized_ds["test"] = tokenized_ds["test"].map(
        lambda e: {'labels': e['label']},  
        batched=True,
        remove_columns=['label']
    )

    tokenized_ds["train"].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_ds["test"].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    print("===========================================================")
    print(tokenized_ds["train"].column_names)
    print("===========================================================")


    ###############################################################################
    # Load model and freeze base parameters
    ###############################################################################

    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))  # Adjust embedding size for new tokens

    for name, param in model.named_parameters():
        if "score" not in name:  # Keep classification head trainable
            param.requires_grad = True

    print(model)

def train_gpt2():
    global model, tokenizer, data_collator, label_names, id2label, label2id, device
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    for name, param in model.named_parameters():
        print(name, param.requires_grad)


    gpt2_trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=CHECKPOINTS, 
            resume_from_checkpoint=True,
            learning_rate=2e-5, 
            per_device_train_batch_size=16, 
            per_device_eval_batch_size=16, 
            num_train_epochs=3, 
            weight_decay=0.01, 
            eval_strategy="epoch",
            save_strategy="epoch", 
            metric_for_best_model="accuracy",  # Change from "eval_loss" to "accuracy"
            load_best_model_at_end=True, 
        ),
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"], 
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics, 
    )

    gpt2_trainer.train()
    gpt2_metrics = gpt2_trainer.evaluate()
    print("Evaluation results before ine-tuning:", gpt2_metrics)
    # gpt2_trainer.save_model(GPT2_FINETUNED_MODEL)  
    # gpt2_trainer.save_state()
    # gpt2_trainer.save_metrics("eval", gpt2_metrics)

    # tokenizer.save_pretrained(GPT2_FINETUNED_MODEL)

###############################################################################
# Performing Parameter-Efficient Fine-Tuning
###############################################################################

import json
def train_lora():
    global model, peft_model, tokenizer, data_collator, label_names, id2label, label2id, device

    # tokenizer = AutoTokenizer.from_pretrained(GPT2_FINETUNED_MODEL)
    # model = AutoModelForSequenceClassification.from_pretrained(GPT2_FINETUNED_MODEL, ignore_mismatched_sizes=True).to(device)
    model.to(device)

    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias="none", task_type=TaskType.SEQ_CLS)
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    lora_trainer = Trainer(
        model=peft_model,  # Make sure to pass the PEFT model here
        args=TrainingArguments(
            output_dir=CHECKPOINTS,
            resume_from_checkpoint=True,
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=3,
            weight_decay=0.01,
            eval_strategy="epoch",
            metric_for_best_model="eval_loss",  
            save_strategy="epoch",
            load_best_model_at_end=True,
            label_names=label_names,
        ),
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    metrics = lora_trainer.evaluate()
    print("Evaluation Metrics:", metrics)
    lora_trainer.train() # resume_from_checkpoint=CHECKPOINTS+"/checkpoint-last")
    lora_metrics = lora_trainer.evaluate()
    print("Evaluation results before fine-tuning:", lora_metrics)

    lora_trainer.save_model(LORA_FINETUNED_MODEL)  
    lora_trainer.save_state()
    lora_trainer.save_metrics("eval", lora_metrics)

    # Save fine-tuned model
    peft_model.save_pretrained(LORA_FINETUNED_MODEL)
    tokenizer.save_pretrained(LORA_FINETUNED_MODEL)  

###############################################################################
# Performing Inference with a PEFT Model
###############################################################################
def train_peft():
    global model, peft_model, tokenizer, data_collator, label_names, id2label, label2id, device
    NUM_LABELS = 2

    model.to(device)
    peft_model = model  # Rename because compatibility 
    # tokenizer = AutoTokenizer.from_pretrained(LORA_FINETUNED_MODEL)
    # peft_model = AutoPeftModelForSequenceClassification.from_pretrained(LORA_FINETUNED_MODEL, num_labels=NUM_LABELS, ignore_mismatched_sizes=True).to(device)
    peft_model.config.pad_token_id = tokenizer.pad_token_id

    hf_trainer = Trainer(
        model=peft_model,  # PEFT model
        args=TrainingArguments(
            output_dir=CHECKPOINTS, 
            resume_from_checkpoint=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss", 
        ),
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # hf_trainer.train(resume_from_checkpoint=CHECKPOINTS+"/checkpoint-last")

    # Save fine-tuned model
    peft_model.save_pretrained(PEFT_FINETUNED_MODEL)
    tokenizer.save_pretrained(PEFT_FINETUNED_MODEL)  

    # Evaluate the fine-tuned model on the test set
    hf_results = hf_trainer.evaluate()
    with open("./evaluation_results.txt", "w") as f:
        f.write(str(hf_results))
    print("Evaluation results for the fine-tuned model:", hf_results)

def p(text, width=80):
    print("\n"*3+"="*width+"\n"+text.upper().center(width)+"\n"+"="*width)

if __name__ == "__main__":
    run_all = len(sys.argv) > 1 and sys.argv[1].lower() == 'run_all'

    if run_all:
        for model_dir in [GPT2_FINETUNED_MODEL, LORA_FINETUNED_MODEL, PEFT_FINETUNED_MODEL]:
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)

    runtime_statistic=""
    start_time = time.time()
    try:
        s = time.time()
        p("load foundation model")
        load_foundation_model()
        runtime_statistic+=f"Time taken to load foundation model: {secToHuman(time.time() - s)}\n"

        if not os.path.exists(GPT2_FINETUNED_MODEL):
            s = time.time()
            p("train gpt2 model")
            train_gpt2()
            runtime_statistic+=f"Time taken to train GPT-2: {secToHuman(time.time() - s)}\n"
        else:
            print(f"Directory {GPT2_FINETUNED_MODEL} already exists. Skipping GPT-2 training.")

        if not os.path.exists(LORA_FINETUNED_MODEL):
            s = time.time()
            p("train lora model")
            train_lora()
            runtime_statistic+=f"Time taken to train LoRA: {secToHuman(time.time() - s)}\n"
        else:
            print(f"Directory {LORA_FINETUNED_MODEL} already exists. Skipping LoRA training.")

        if not os.path.exists(PEFT_FINETUNED_MODEL):
            s = time.time()
            p("train peft model")
            train_peft()
            end_time = time.time()
            runtime_statistic+=f"Time taken to train PEFT: {secToHuman(time.time() - s)}\n"
        else:
            with open(os.path.join(PEFT_FINETUNED_MODEL, "evaluation_results.txt"), "r") as f:
                print(f.read())
    except Exception as e:
        print(traceback.format_exc())
    finally:
        end_time = time.time()
        elapsed_time = secToHuman(end_time - start_time)
        print(f"{runtime_statistic}Runtime sum: {elapsed_time}")
