"""Flares model service.
This module discovers linguistic models stored under models/flares,
loads tokenizers and transformers, and exposes dynamic prediction helpers
for token classification and sequence classification models.
"""

import os
import json
from typing import List, Dict
import torch
from fastapi import HTTPException
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoConfig
from src.schemas.schemas import ReliabilitySample, ReliabilityPrediction, Tag, TextRequest
from src.utils.flares.train_models import extract_spans

# Label mappings
WH_LABELS = ["O", "B-WHO", "I-WHO", "B-WHAT", "I-WHAT", "B-WHEN", "I-WHEN",
             "B-WHERE", "I-WHERE", "B-WHY", "I-WHY", "B-HOW", "I-HOW"]
WH_LABEL_TO_ID = {label: i for i, label in enumerate(WH_LABELS)}
ID_TO_WH_LABEL = {i: label for label, i in WH_LABEL_TO_ID.items()}

RELIABILITY_LABELS = ["confiable", "semiconfiable", "no confiable"]
RELIABILITY_LABEL_TO_ID = {label: i for i, label in enumerate(RELIABILITY_LABELS)}
ID_TO_RELIABILITY_LABEL = {i: label for label, i in RELIABILITY_LABEL_TO_ID.items()}

class FlaresModels:
    def __init__(self):
        self.models: Dict[str, object] = {}
        self.tokenizers: Dict[str, object] = {}
        self.model_types: Dict[str, str] = {} 
        self.load_models()

    def load_models(self):
        model_dir = "models/flares"
        
        if not os.path.exists(model_dir):
            print(f"[WARNING] Model directory {model_dir} does not exist. No flares models loaded.")
            return

        loaded = False
        for model_name in os.listdir(model_dir):
            model_path = os.path.join(model_dir, model_name)
            if not os.path.isdir(model_path):
                continue
                
            # Skip hidden directories
            if model_name.startswith('.'):
                continue
                
            # Try to load model and tokenizer
            try:
                config_path = os.path.join(model_path, "config.json")
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)

                # Determine model type by checking for config files or directory names
                if "5w1h" in model_name:
                    model_type = "5w1h"  
                    model = AutoModelForTokenClassification.from_pretrained(model_path)
                else:
                    model_type = "reliability" 
                    model = AutoModelForSequenceClassification.from_pretrained(model_path)
                
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
                self.model_types[model_name] = model_type
                print(f"[INFO] Loaded flares model: {model_name} ({model_type})")
                loaded = True
                
            except Exception as e:
                print(f"[ERROR] Failed to load model {model_name} from {model_path}: {str(e)}")
                continue

        if not loaded:
            print(f"[WARNING] No flares models loaded from {model_dir}. Ensure models exist and are valid.")

    def list_models(self) -> List[str]:
        return list(self.models.keys())

    def get_model_type(self, model_name: str) -> str:
        if model_name not in self.model_types:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not loaded")
        return self.model_types[model_name]

    

    def predict_wh_batch(
        self,
        model_name: str,
        texts: List[TextRequest],
        max_length: int = 128
    ) -> List[List[Tag]]:

        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        device = next(model.parameters()).device

        raw_texts = [t.Text for t in texts]

        encodings = tokenizer(
            raw_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        offset_mapping = encodings.pop("offset_mapping")

        encodings = {
            k: v.to(device)
            for k, v in encodings.items()
        }

        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2).cpu().tolist()

        results = []

        for text_req, pred_seq, offsets in zip(
            texts,
            predictions,
            offset_mapping.tolist()
        ):

            spans = extract_spans(
                text_req.Text,
                pred_seq,
                offsets
            )

            tags = [
                Tag(
                    Tag_Start=span["start"],
                    Tag_End=span["end"],
                    Label_5W1H=span["label"],
                    Tag_Text=span["text"]
                )
                for span in spans
            ]

            results.append(tags)

        return results


    def predict_reliability_batch(
        self,
        model_name: str,
        records: List[ReliabilitySample],
        max_length: int = 512
    ) -> List[ReliabilityPrediction]:

        if model_name not in self.models:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} not loaded"
            )

        if not records:
            return []

        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        device = next(model.parameters()).device

        texts = [record.Tag_Text for record in records]

        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )

        encodings = {
            k: v.to(device)
            for k, v in encodings.items()
        }

        with torch.no_grad():
            outputs = model(**encodings)
            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()

        results: List[ReliabilityPrediction] = []

        for record, pred in zip(records, preds):

            results.append(
                ReliabilityPrediction(
                    Id=record.Id,
                    Text=record.Text,
                    Label_5W1H=record.Label_5W1H,
                    Tag_Text=record.Tag_Text,
                    Tag_Start=record.Tag_Start,
                    Tag_End=record.Tag_End,
                    Reliability_Label=ID_TO_RELIABILITY_LABEL[pred],
                )
            )

        return results

    def predict_batch(self, model_name: str, payload: list, max_length: int = 512):
        """Predict a batch of requests for a loaded flares model.

        The payload type depends on the model type: token classification models
        expect a list of TextRequest objects; reliability models expect a list of
        ReliabilitySample objects.
        """
        model_type = self.get_model_type(model_name)
        if model_type == "wh":
            return self.predict_wh_batch(model_name, payload, max_length=max_length)
        return self.predict_reliability_batch(model_name, payload, max_length=max_length)

# Initialize the models
flares_models = FlaresModels()
