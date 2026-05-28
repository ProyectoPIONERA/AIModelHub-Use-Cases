"""
Evaluation script for Flares linguistic tasks: 5W1H token classification and reliability classification.
"""

import argparse
import json
import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score
import numpy as np
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support
import re

# Importar lógica compartida desde train_models.py
from src.utils.flares.train_models import (
    WH_LABELS, WH_LABEL_TO_ID, ID_TO_WH_LABEL,
    RELIABILITY_LABELS, RELIABILITY_LABEL_TO_ID, ID_TO_RELIABILITY_LABEL,
    STOPWORDS,
    WHDataset, ReliabilityDataset,
    load_tokenizer,
    compute_wh_metrics, compute_reliability_metrics,
    extract_spans, decode_bio_spans, reconstruct_spans, is_valid_span, clean_spans,
    IGNORE_INDEX
)

# Configuration
SUBTASK1_TEST_FILE = "./data/flares-datasets/5w1h_subtarea_1_test.json"
SUBTASK2_TEST_FILE = "./data/flares-datasets/5w1h_subtarea_2_test.json"
BATCH_SIZE = 4
MAX_LENGTH = 128
SEED = 42
MODELS_DIR='./models/flares'
# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


def parse_args():
    """
    Script input parameters.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate BERT models for 5W1H and reliability classification"
    )
    parser.add_argument("--task", default='5W1H', type=str, help="Path to WH test JSONL file")
    parser.add_argument("--model-name", required = True, help="Model to evaluate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    
    return parser.parse_args()


def load_data(data_path: str) -> List[Dict]:
    """Load JSON data from file"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(len(data))
    return data


def evaluate_wh_model(
    test_data: List[Dict],
    model_dir: str,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    use_cpu: bool = False,
    fp16: bool = False,
):
    print(f"Evaluating 5W1H model from {model_dir}...")

    tokenizer = load_tokenizer(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    device = "cuda" if torch.cuda.is_available() and not use_cpu else "cpu"
    model.to(device)

    eval_dataset = WHDataset(test_data, tokenizer, max_length=max_length)

    training_args = TrainingArguments(
        output_dir=model_dir,
        per_device_eval_batch_size=batch_size,
        fp16=fp16,
        use_cpu=use_cpu,
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=eval_dataset,
        compute_metrics=lambda eval_pred: compute_wh_metrics(
            eval_dataset,
            eval_pred,
            model_dir
        ),
    )

    results = trainer.evaluate()

    print("\n5W1H Evaluation Results:")
    print(f"  Precision: {results.get('eval_precision', 0):.4f}")
    print(f"  Recall:    {results.get('eval_recall', 0):.4f}")
    print(f"  F1:        {results.get('eval_f1', 0):.4f}")

    return results


def evaluate_reliability_model(
    test_data: List[Dict],
    model_dir: str,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    use_cpu: bool = False,
    fp16: bool = False,
):
    """Evaluate reliability classification model"""
    print(f"Evaluating reliability model from {model_dir}...")

    tokenizer = load_tokenizer(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        ignore_mismatched_sizes=True,
    )
    model.eval()

    # Reutilizar ReliabilityDataset directamente
    eval_dataset = ReliabilityDataset(test_data, tokenizer, max_length=max_length)

    # Preparar TrainingArguments para evaluación
    training_args = TrainingArguments(
        output_dir=model_dir,
        per_device_eval_batch_size=batch_size,
        fp16=fp16,
        use_cpu=use_cpu,
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
         eval_dataset=eval_dataset,
        compute_metrics=lambda eval_pred: compute_reliability_metrics(
            eval_pred,
            model_dir
        )
    )

    results = trainer.evaluate()

    print(results)
    print(f"Reliability Evaluation Results:")
    print(f"  Accuracy: {results.get('eval_accuracy',0):.4f}")
    print(f"  Precision: {results.get('eval_precision',0):.4f}")
    print(f"  Recall: {results.get('eval_recall',0):.4f}")
    print(f"  F1: {results.get('eval_f1',0):.4f}")

    return results


def main():
    """Main evaluation function"""
    args = parse_args()

    use_cpu = args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available())
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        use_cpu = True

    if args.fp16 and use_cpu:
        print("FP16 requested but CPU is selected; disabling fp16.")
        args.fp16 = False

    # Evaluate 5W1H model
    if args.task == '5W1H':
        print("\n=== 5W1H Evaluation ===")
        test_data = load_data(SUBTASK1_TEST_FILE)
        evaluate_wh_model(
            test_data,
            os.path.join(MODELS_DIR,args.model_name),
            batch_size=args.batch_size,
            max_length=args.max_length,
            use_cpu=use_cpu,
            fp16=args.fp16,
        )
    else:
    # Evaluate reliability model
        print("\n=== Reliability Evaluation ===")
        test_data = load_data(SUBTASK2_TEST_FILE)
        evaluate_reliability_model(
            test_data,
            os.path.join(MODELS_DIR,args.model_name),
            batch_size=args.batch_size,
            max_length=args.max_length,
            use_cpu=use_cpu,
            fp16=args.fp16,
        )

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()