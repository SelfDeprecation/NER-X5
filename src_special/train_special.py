import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification
)
from sklearn.model_selection import train_test_split

from io_helpers import read_train_csv
from dataset import prepare_labels_mapping, build_hf_tokenized_dataset
from utils import compute_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    set_seed(args.seed)

    df = read_train_csv(args.train_csv)
    samples = df['sample'].tolist()
    annotations = df['annotation_parsed'].tolist()

    label_to_id, id_to_label = prepare_labels_mapping(annotations)

    train_s, val_s, train_a, val_a = train_test_split(
        samples, annotations, test_size=0.1, random_state=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, add_prefix_space=True)
    train_dataset = build_hf_tokenized_dataset(
        train_s, train_a, tokenizer, label_to_id,
        max_length=args.max_length,
        label_all_tokens=args.label_all_tokens
    )
    val_dataset = build_hf_tokenized_dataset(
        val_s, val_a, tokenizer, label_to_id,
        max_length=args.max_length,
        label_all_tokens=args.label_all_tokens
    )

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=len(label_to_id))
    model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, config=config)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.logging_steps,
        save_steps=args.logging_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=args.seed,
        fp16=True,
    )

    def compute_metrics_wrapper(p):
        p.id_to_label = {v: k for k, v in label_to_id.items()}
        return compute_metrics(p)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="outputs/run")
    parser.add_argument("--output_dir", type=str, default="outputs/run")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--label_all_tokens", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=200)
    args = parser.parse_args()

    main(args)
