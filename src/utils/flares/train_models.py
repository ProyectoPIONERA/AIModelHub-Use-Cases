"""
Training models script for Flares linguistic tasks: 5W1H token classification and reliability classification.
"""

import argparse
import json
import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    BertTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from transformers import Trainer
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_recall_fscore_support
import re

# Configuration
SUBTASK1_TRAIN_FILE = "./data/flares-datasets/5w1h_subtarea_1_train.json"
SUBTASK2_TRAIN_FILE = "./data/flares-datasets/5w1h_subtarea_2_train.json"
MODEL_SAVE_DIR = "./models/flares"
BATCH_SIZE = 4
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
SEED = 42
GRADIENT_ACCUMULATION_STEPS = 3
WARMUP_RATIO = 0.1
IGNORE_INDEX = -100

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Label mappings
WH_LABELS = ["O", "B-WHO", "I-WHO", "B-WHAT", "I-WHAT", "B-WHEN", "I-WHEN",
             "B-WHERE", "I-WHERE", "B-WHY", "I-WHY", "B-HOW", "I-HOW"]
WH_LABEL_TO_ID = {label: i for i, label in enumerate(WH_LABELS)}
ID_TO_WH_LABEL = {i: label for label, i in WH_LABEL_TO_ID.items()}

RELIABILITY_LABELS = ["confiable", "semiconfiable", "no confiable"]
RELIABILITY_LABEL_TO_ID = {label: i for i, label in enumerate(RELIABILITY_LABELS)}
ID_TO_RELIABILITY_LABEL = {i: label for label, i in RELIABILITY_LABEL_TO_ID.items()}

STOPWORDS = {
    "de", "la", "el", "los", "las",
    "y", "o", "en", "un", "una"
}

def parse_args():
    """
    Script input parameters.
    """
    parser = argparse.ArgumentParser(
        description="Train BERT models for 5W1H and reliability classification"
    )
    parser.add_argument("--wh-train-file", default=SUBTASK1_TRAIN_FILE,type=str,help="Path to WH training JSONL file")
    parser.add_argument("--reliability-train-file",default=SUBTASK2_TRAIN_FILE, type=str, help="Path to reliability training JSONL file")
    parser.add_argument("--output-dir", default=MODEL_SAVE_DIR,help="Directory to save trained models")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--model", required=True,help="Base transformer model name")
    parser.add_argument("--train-both",default=True, action="store_true",help="Train both WH and reliability models")
    parser.add_argument("--train-wh-only",default=False, action="store_true",help="Train only WH model")
    parser.add_argument("--train-reliability-only", default=False, action="store_true",help="Train only reliability model")
    
    return parser.parse_args()


# =========================================================
# Custom Pytorch Datasets Classes
# =========================================================

class WHDataset(Dataset):
    """
    Dataset class for 5W1H task
    """
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.encodings = []
        self.labels = []
        self.offsets = []
        self.texts = []
        for item in data:

            enc = tokenizer(
                item["Text"],
                truncation=True,
                padding=False,
                max_length=max_length,
                return_offsets_mapping=True
            )

            offsets = enc["offset_mapping"]
            enc.pop("offset_mapping")

            labels = [WH_LABEL_TO_ID["O"]] * len(offsets)

            for tag in item["Tags"]:
                start_char = tag["Tag_Start"]
                end_char = tag["Tag_End"]
                entity = tag["5W1H_Label"]

                first = True
                for i, (s, e) in enumerate(offsets):

                    if e <= start_char or s >= end_char:
                        continue

                    if first:
                        labels[i] = WH_LABEL_TO_ID[f"B-{entity}"]
                        first = False
                    else:
                        labels[i] = WH_LABEL_TO_ID[f"I-{entity}"]

            self.encodings.append(enc)
            self.labels.append(labels)
            self.offsets.append(offsets)
            self.texts.append(item["Text"])

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):

        enc = self.encodings[idx]
        labels = self.labels[idx]

        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels": torch.tensor(labels)
        }
    



class ReliabilityDataset(Dataset):
    """
    Dataset for realibility classification task
    """
    def __init__(self, data, tokenizer, max_length=MAX_LENGTH):

        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for item in data:
            self.samples.append({
                "text": item["Tag_Text"],
                "label": RELIABILITY_LABEL_TO_ID[item["Reliability_Label"]]
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        encoding = self.tokenizer(
            sample["text"],
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )

        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": torch.tensor(sample["label"]),
        }


def load_data(data_path: str) -> List[Dict]:
    """Load JSON data from file"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data




# =========================================================
# 5W1H SPAN EXTRACTION
# =========================================================

def decode_bio_spans(label_seq, id_to_label):
    """
    Decode BIO tags sequences to spans (start, end, label).
    """
    spans = []

    current_start = None
    current_end = None
    current_label = None

    for i, label_id in enumerate(label_seq):

        if label_id == IGNORE_INDEX:
            continue

        label = id_to_label.get(int(label_id), "O")

        # -------------------------------------------------
        # BEGIN
        # -------------------------------------------------
        if label.startswith("B-"):

            if current_start is not None:
                spans.append(
                    (current_start, current_end, current_label)
                )

            current_start = i
            current_end = i
            current_label = label[2:]

        # -------------------------------------------------
        # INSIDE
        # -------------------------------------------------
        elif label.startswith("I-"):

            entity_type = label[2:]

            # valid continuation
            if (
                current_start is not None
                and current_label == entity_type
            ):
                current_end = i

            # orphan I-
            else:

                if current_start is not None:
                    spans.append(
                        (current_start, current_end, current_label)
                    )

                current_start = i
                current_end = i
                current_label = entity_type

        # -------------------------------------------------
        # OUTSIDE
        # -------------------------------------------------
        else:

            if current_start is not None:

                spans.append(
                    (current_start, current_end, current_label)
                )

                current_start = None
                current_end = None
                current_label = None

    # close final span
    if current_start is not None:

        spans.append(
            (current_start, current_end, current_label)
        )

    return spans


def reconstruct_spans(text, spans, offsets):
    reconstructed = []
    for start_tok, end_tok, entity_type in spans:

        char_start = None
        char_end = None

        for i in range(start_tok, end_tok + 1):
            if i >= len(offsets):
                break  # IMPORTANT: stop, don't continue

            s, e = offsets[i]
            if s == e:
                continue

            if char_start is None:
                char_start = s

            char_end = e # always update to last valid token

        if(char_start != None and char_end != None):
            span_text = text[char_start:char_end]  # ✅ correct slicing
            reconstructed.append({
                "start": char_start,
                "end": char_end,
                "label": entity_type,
                "text": span_text.strip(),
            })

    return reconstructed


def is_valid_span(span_text):

    if not span_text:
        return False

    text = span_text.strip()

    # punctuation only
    if re.fullmatch(r"[\W_]+", text):
        return False

    # wordpiece leftovers
    if "##" in text:
        return False

    # too short
    if len(text) <= 1:
        return False

    # stopword only
    if text.lower() in STOPWORDS:
        return False

    # excessive punctuation
    punct_count = sum(
        1 for c in text
        if not c.isalnum()
    )

    if punct_count / max(len(text), 1) > 0.5:
        return False

    return True

def clean_spans(spans):

    cleaned = []

    for span in spans:

        if not is_valid_span(span["text"]):
            continue

        cleaned.append(span)

    return cleaned

def extract_spans(text,label_seq,offsets):

    # -----------------------------------------------
    # TOKEN-LEVEL BIO DECODE
    # -----------------------------------------------
    token_spans = decode_bio_spans(
        label_seq,
        ID_TO_WH_LABEL
    )

    # -----------------------------------------------
    # RECONSTRUCT REAL TEXT SPANS
    # -----------------------------------------------
    spans = reconstruct_spans(
        text=text,
        spans=token_spans,
        offsets=offsets,
    )


    # -----------------------------------------------
    # CLEAN SPANS
    # -----------------------------------------------
    return clean_spans(spans)

# =========================================================
# FLARES OFFICIAL-STYLE METRICS
# =========================================================

def compute_wh_metrics(dataset,eval_pred, output_dir=None):
    """
    FLARES Subtask 1 metrics:
    Correct / Partial / Missing / Spurious
    """

    predictions, labels = eval_pred

     # logits -> softmax argmax
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.argmax(predictions, axis=-1)

    correct = 0
    partial = 0
    spurious = 0
    missing = 0

    batch_offsets = dataset.offsets
    batch_texts = dataset.texts

    for pred_seq, gold_seq, offsets, text in zip(
        predictions,
        labels,
        batch_offsets,
        batch_texts
    ):

        pred_spans = extract_spans(text,pred_seq,offsets)
        gold_spans = extract_spans(text,gold_seq,offsets)


        matched_gold = set()

        for span in pred_spans:
            p_start = span["start"]
            p_end = span["end"]
            p_type = span["label"]

            found_match = False

            for idx, gold_span in enumerate(gold_spans):
                g_start = gold_span["start"]
                g_end = gold_span["end"]
                g_type = gold_span["label"]

                if idx in matched_gold:
                    continue

                # Exact match
                if (
                    p_start == g_start
                    and p_end == g_end
                    and p_type == g_type
                ):
                    correct += 1
                    matched_gold.add(idx)
                    found_match = True
                    break

                # Partial overlap
                overlap = not (p_end < g_start or p_start > g_end)

                if overlap and p_type == g_type:
                    partial += 1
                    matched_gold.add(idx)
                    found_match = True
                    break

            if not found_match:
                spurious += 1

        missing += len(gold_spans) - len(matched_gold)

    precision = (correct + 0.5 * partial) / max(correct + partial + spurious, 1)
    recall = (correct + 0.5 * partial) / max(correct + partial + missing, 1)

    f1 = (2 * precision * recall) / max(precision + recall, 1e-8)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": correct,
        "partial": partial,
        "missing": missing,
        "spurious": spurious,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        metrics_file = os.path.join(output_dir, "wh_metrics.jsonl")
        with open(metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    return metrics


def compute_reliability_metrics(eval_pred, output_dir=None):
    """
    Subtask 2:
    Accuracy, Precision, Recall, F1
    """

    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        metrics_file = os.path.join(output_dir, "reliability_metrics.jsonl")
        with open(metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    return metrics


def save_wh_metrics_to_jsonl(metrics: Dict, output_dir: str):
    """
    Save 5W1H metrics to json file
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, "wh_metrics.jsonl")
    with open(metrics_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")


def save_reliability_metrics_to_jsonl(metrics: Dict, output_dir: str):
    """
    Save reliability metrics to json file
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, "reliability_metrics.jsonl")
    with open(metrics_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

# =========================================================
# TRAIN
# =========================================================


def load_tokenizer(model_name: str):
    """Load a tokenizer, falling back to a compatible slow tokenizer if needed."""
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        try:
            return AutoTokenizer.from_pretrained(model_name, use_fast=False)
        except Exception:
            try:
                return BertTokenizer.from_pretrained(model_name)
            except Exception as exc:
                raise ValueError(
                    f"Unable to load tokenizer for {model_name}. "
                    "Install sentencepiece or tiktoken if required, or use a compatible tokenizer model. "
                    f"Original error: {exc}"
                )


def train_wh_model(
    train_data: List[Dict],
    val_data: List[Dict],
    model_name: str,
    output_dir: str = f"{MODEL_SAVE_DIR}/wh_model",
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    learning_rate: float = LEARNING_RATE,
    fp16: bool = False,
    use_cpu: bool = False,
    num_epochs: int = NUM_EPOCHS,
):
    """Train 5W1H token classification model"""
    print(f"Training 5W1H model with {model_name}...")

    tokenizer = load_tokenizer(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(WH_LABELS),
        id2label=ID_TO_WH_LABEL,
        label2id=WH_LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )

    train_dataset = WHDataset(train_data, tokenizer, max_length=max_length)
    val_dataset = WHDataset(val_data, tokenizer, max_length=max_length)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=fp16,
        use_cpu=use_cpu,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=WARMUP_RATIO,
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_wh_metrics(val_dataset, eval_pred, output_dir),

    )

    trainer.train()
    # Save final model and tokenizer directly to output_dir (no /final subfolder)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
   

    return trainer

def train_reliability_model(
    train_data: List[Dict],
    val_data: List[Dict],
    model_name: str,
    output_dir: str = f"{MODEL_SAVE_DIR}/reliability_model",
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    learning_rate: float = LEARNING_RATE,
    fp16: bool = False,
    use_cpu: bool = False,
    num_epochs: int = NUM_EPOCHS,
):
    """Train reliability classification model"""
    print(f"Training reliability model with {model_name}...")

    tokenizer = load_tokenizer(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(RELIABILITY_LABELS),
        id2label=ID_TO_RELIABILITY_LABEL,
        label2id=RELIABILITY_LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )

    train_dataset = ReliabilityDataset(train_data, tokenizer, max_length=max_length)
    val_dataset = ReliabilityDataset(val_data, tokenizer, max_length=max_length)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=fp16,
        use_cpu=use_cpu,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=WARMUP_RATIO,
        report_to="none"
    )

    data_collator = DataCollatorWithPadding(tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_reliability_metrics(eval_pred, output_dir),
    )

    trainer.train()
    # Save final model and tokenizer directly to output_dir (no /final subfolder)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer

def main():
    """Main training function"""
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run_wh=False
    run_reliability=False

    if(args.train_wh_only):
        run_wh=True
    elif(args.train_reliability_only):
        run_reliability=True
    else:
        run_wh=True
        run_reliability=True

    use_cpu = args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available())
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        use_cpu = True

    if args.fp16 and use_cpu:
        print("FP16 requested but CPU is selected; disabling fp16.")
        args.fp16 = False

    if run_wh:
        print("\n=== 5W1H Training ===")
        # Load training data
        print("Loading training data...")
        train_data = load_data(args.wh_train_file)
        # Split into train/validation
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=SEED)
        print(f"Train: {len(train_data)}, Validation: {len(val_data)}")
        
        # Save model with format model_name-5w1h
        model_name_base = args.model.replace("/", "-")
        primary_wh_dir = os.path.join(args.output_dir, f"{model_name_base}-5w1h")
        train_wh_model(
            train_data,
            val_data,
            model_name=args.model,
            output_dir=primary_wh_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
            learning_rate=args.learning_rate,
            fp16=args.fp16,
            use_cpu=use_cpu,
            num_epochs=args.epochs,
        )


    if run_reliability:
        print("\n=== Reliability Training ===")
        # Load training data
        print("Loading training data...")
        train_data = load_data(args.reliability_train_file)
        # Split into train/validation
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=SEED)
        print(f"Train: {len(train_data)}, Validation: {len(val_data)}")
        
        # Save model with format model_name-reliability
        model_name_base = args.model.replace("/", "-")
        primary_rel_dir = os.path.join(args.output_dir, f"{model_name_base}-reliability")
        train_reliability_model(
            train_data,
            val_data,
            model_name=args.model,
            output_dir=primary_rel_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
            learning_rate=args.learning_rate,
            fp16=args.fp16,
            use_cpu=use_cpu,
            num_epochs=args.epochs,
        )


    print("\nTraining completed!")
    print(f"Models saved to {args.output_dir}")


if __name__ == "__main__":
    main()
