# src/services/mobility_service.py

"""Mobility model service.

This module discovers mobility models stored under models/mobility,
loads preprocessing configs and trained models,
and exposes dynamic prediction helpers.
"""

import json
import os
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.preprocessing import StandardScaler

from src.utils.mobility.train_models import encode_stop_series


class MobilityModels:
    def __init__(self):
        self._preprocess_cache: Dict[str, Dict[str, Any]] = {}
        self._scaler_cache: Dict[str, Any] = {}

        # cache[target][model_name]
        self._models_cache: Dict[str, Dict[str, Any]] = {}

        self._discover_models()

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------

    def _discover_models(self):

       
        self._models_by_target: Dict[str, List[str]] = {}

        models_dir = "models/mobility"

        if not os.path.exists(models_dir):
            print(
                "[WARNING] Model directory models/mobility "
                "does not exist."
            )
            return

        for filename in os.listdir(models_dir):

        
            if not filename.endswith("_model.pkl"):
                continue

            base = filename[:-len("_model.pkl")]

            parts = base.split("_", 1)

            if len(parts) != 2:
                continue

            model_name = parts[0]
            target = parts[1]

            config_path = os.path.join(
                models_dir,
                f"{target}_config.json",
            )

            if not os.path.exists(config_path):
                continue

            self._models_by_target.setdefault(
                target,
                []
            ).append(model_name)

    # ------------------------------------------------------------------
    # preprocess config
    # ------------------------------------------------------------------

    def _get_preprocess(self, target: str) -> Dict[str, Any]:

        if target not in self._preprocess_cache:

            config_path = (
                f"models/mobility/{target}_config.json"
            )

            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"Config file not found: {config_path}"
                )

            with open(config_path, "r") as f:
                self._preprocess_cache[target] = json.load(f)

        return self._preprocess_cache[target]


    # ------------------------------------------------------------------
    # model loading
    # ------------------------------------------------------------------

    def _split_model_name(self, model_name: str):

        parts = model_name.split("_", 1)

        if len(parts) != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model name: {model_name}",
            )

        model_base = parts[0]
        target = parts[1]

        return model_base, target

    def _get_models(self, target: str):

        if target not in self._models_cache:

            models = {}

            for model_name in self._models_by_target.get(
                target,
                [],
            ):

                model_path = (
                    f"models/mobility/"
                    f"{model_name}_{target}_model.pkl"
                )

                if os.path.exists(model_path):
                    models[model_name] = joblib.load(
                        model_path
                    )

            self._models_cache[target] = models

        return self._models_cache[target]

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    def get_model_columns(
        self,
        model_name: str,
    ) -> List[str]:

        _, target = self._split_model_name(
            model_name
        )

        preprocess = self._get_preprocess(target)

        categorical = preprocess.get(
            "categorical_cols",
            [],
        )

        numeric = preprocess.get(
            "numeric_cols",
            [],
        )

        return categorical + numeric

    def list_models(self) -> List[str]:

        result = []

        for target, model_names in (
            self._models_by_target.items()
        ):
            for model_name in model_names:
                result.append(
                    f"{model_name}_{target}"
                )

        return result

    # ------------------------------------------------------------------
    # preprocess
    # ------------------------------------------------------------------

    def _preprocess_data(
        self,
        df: pd.DataFrame,
        target: str,
    ) -> pd.DataFrame:

        preprocess = self._get_preprocess(target)


        categorical_cols = preprocess[
            "categorical_cols"
        ]

        numeric_cols = preprocess[
            "numeric_cols"
        ]

        stop_to_idx = preprocess[
            "stop_mapping"
        ]

        for col in categorical_cols:

            if col in [
                "from_stop_id",
                "to_stop_id",
            ]:

                df[col] = encode_stop_series(
                    df[col],
                    stop_to_idx,
                )

            else:

                mapper = preprocess.get(
                    f"{col}_mapping"
                )

                if mapper is None:
                    raise ValueError(
                        f"Missing mapping for column "
                        f"'{col}' in preprocessing config."
                    )

                df[col] = (
                    df[col]
                    .astype(str)
                    .map(mapper)
                    .fillna(-1)
                    .astype(int)
                )
                
        return df

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        model_name: str,
        samples: List[Dict[str, Any]],
    ) -> List[float]:

        model_base, target = (
            self._split_model_name(model_name)
        )

        models = self._get_models(target)

        if model_base not in models:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} not loaded",
            )

        model = models[model_base]

        df = pd.DataFrame(samples)

        df = self._preprocess_data(
            df,
            target,
        )

        features = self._get_preprocess(
            target
        )["features"]

        predictions = model.predict(
            df[features]
        )

        return predictions.tolist()


# singleton
mobility_models = MobilityModels()