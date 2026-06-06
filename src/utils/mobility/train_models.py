import pandas as pd
import numpy as np
import json
import joblib
import os
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import GroupShuffleSplit



DEFAULT_CATEGORIC_COLUMNS = ['trip_id', 'from_stop_id', 'to_stop_id', 'route_id']
DEFAULT_NUMERIC_COLUMNS = ['scheduled_travel_time','shape_distance', 'is_peak', 'hour_sin', 'hour_cos', 'weekday_sin',
       'weekday_cos', 'previous_delay_ratio','previous_delay_delta']

TARGET_OPTIONS=["actual_travel_time","delay","previous_delay"]
DEFAULT_TARGET = "actual_travel_time"


# =========================
# Parse arguments
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Train mobility models for travel time prediction")
    parser.add_argument("--data-path", default="./data/mobility-datasets/segments_train.csv", type=str)
    parser.add_argument("--test-path", default="./data/mobility-datasets/segments_test.csv", type=str)
    parser.add_argument("--output-dir", default="./models/mobility/", type=str)
    parser.add_argument("--categorical-cols", nargs="+", default=DEFAULT_CATEGORIC_COLUMNS, choices=DEFAULT_CATEGORIC_COLUMNS)
    parser.add_argument("--numeric-cols", nargs="+", default=DEFAULT_NUMERIC_COLUMNS,choices=DEFAULT_NUMERIC_COLUMNS)
    parser.add_argument("--target", default=DEFAULT_TARGET, type=str, choices=TARGET_OPTIONS)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--lightgbm-n-estimators", type=int, default=2000)
    parser.add_argument("--lightgbm-learning-rate", type=float, default=0.03)
    parser.add_argument("--lightgbm-num-leaves", type=int, default=31)
    parser.add_argument("--lightgbm-min-child-samples", type=int, default=50)
    parser.add_argument("--lightgbm-subsample", type=float, default=0.85)
    parser.add_argument("--lightgbm-colsample-bytree", type=float, default=0.85)
    parser.add_argument("--rf-n-estimators", type=int, default=700)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=5)
    parser.add_argument("--rf-max-features", type=parse_max_features, default=0.8)
    parser.add_argument("--ct-max-depth", type=int, default=6)
    parser.add_argument("--ct-max-iter", type=int, default=2500)
    parser.add_argument("--ct-learning-rate", type=float, default=0.03)
    parser.add_argument("--ct-l2-leaf-reg", type=float, default=6.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    return parser.parse_args()

def encode_stop_series(series, mapping):
        return (
            series.astype(str)
            .map(mapping)
            .fillna(-1)
            .astype(int)
        )


def evaluate(model, X, y, name=""):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    print(f"{name} -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
    return mae, rmse, r2

def clean_outliers(df,column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_clean = df[
        df[column].between(lower, upper)
    ]
    return df_clean

# =========================
# Main training logic
# =========================


def main():
    args = parse_args()

    # === 1. Load data ===
    df = pd.read_csv(args.data_path)
    test_df = pd.read_csv(args.test_path)

    
       # === 2. Feature selection ===
    target = args.target
    categorical_cols = args.categorical_cols
    numeric_cols = args.numeric_cols

    if(target == 'previous_delay'):
        numeric_cols = [x for x in numeric_cols if x not in ['previous_delay', 'previous_delay_ratio','previous_delay_delta']]
    

    features = categorical_cols + numeric_cols

    

    # === 3. Train/validation split ===
    gss = GroupShuffleSplit(
            test_size=args.test_size,
            random_state=args.random_state
        )

    train_idx, test_idx = next(
        gss.split(df, groups=df["journey_id"])
    )

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[test_idx].copy()


    # === 4. Preprocessing (fit on train only) ===
    stops_df = pd.read_csv("./data/mobility-datasets/GTFS_Scheduled/stops.txt")
    all_stops = stops_df["stop_id"].astype(str).unique()
    stop_to_idx = {stop_id: i for i, stop_id in enumerate(all_stops)}

    preprocess = {
        "stop_mapping": stop_to_idx,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "features": features
    }

    # Encode categoricals
    for col in categorical_cols:
        if col in ["from_stop_id", "to_stop_id"]:
            train_df[col] = encode_stop_series(train_df[col], stop_to_idx)
            val_df[col] = encode_stop_series(val_df[col], stop_to_idx)
        else:
            unique_vals = train_df[col].astype(str).unique()
            mapper = {v: i for i, v in enumerate(unique_vals)}
            train_df[col] = train_df[col].astype(str).map(mapper).fillna(-1).astype(int)
            val_df[col] = val_df[col].astype(str).map(mapper).fillna(-1).astype(int)
            preprocess[col + "_mapping"] = mapper


    # Save preprocessing config
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{args.target}_config.json"), "w") as f:
        json.dump(preprocess, f, indent=4)

    # === 5. Prepare datasets ===
    X_train = train_df[features]
    y_train = train_df[target]

    X_val = val_df[features]
    y_val = val_df[target]
    cat_feature_indices = [
        X_train.columns.get_loc(col)
        for col in categorical_cols
        if col in X_train.columns
    ]

    # === 6. Models ===
    models = {
        "lightgbm": LGBMRegressor(
            n_estimators=args.lightgbm_n_estimators,
            learning_rate=args.lightgbm_learning_rate,
            num_leaves=args.lightgbm_num_leaves,
            min_child_samples=args.lightgbm_min_child_samples,
            subsample=args.lightgbm_subsample,
            subsample_freq=1,
            colsample_bytree=args.lightgbm_colsample_bytree,
            reg_alpha=0.1,
            reg_lambda=1.0,
            max_depth=-1,
            random_state=args.random_state,
            objective="regression",
            metric="rmse",
            n_jobs=-1,
            verbosity=-1
        ),
        "randomforest": RandomForestRegressor(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            max_features=args.rf_max_features,
            n_jobs=-1,
            criterion="squared_error",
            random_state=args.random_state
        ),
        "catboost": CatBoostRegressor(
            iterations=args.ct_max_iter,
            learning_rate=args.ct_learning_rate,
            depth=args.ct_max_depth,
            l2_leaf_reg=args.ct_l2_leaf_reg,
            loss_function="RMSE",
            eval_metric="R2",
            random_seed=args.random_state,
            allow_writing_files=False,
            verbose=False
        )
    }

   
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        if name == "lightgbm":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                categorical_feature=categorical_cols,
                callbacks=[
                    early_stopping(args.early_stopping_rounds, verbose=False),
                    log_evaluation(period=0),
                ],
            )
        elif name == "catboost":
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_feature_indices,
                early_stopping_rounds=args.early_stopping_rounds,
                use_best_model=True,
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)
        print("Validation performance:")
        mae, rmse, r2 = evaluate(model, X_val, y_val, name)
        joblib.dump(model, os.path.join(args.output_dir, f"{name}_{args.target}_model.pkl"))
        results[name] = {"mae": mae, "rmse": rmse, "r2": r2}

    # === 8. Test evaluation ===
    # Apply same preprocessing (categoricals)
    for col in categorical_cols:
        if col in ["from_stop_id", "to_stop_id"]:
            test_df[col] = encode_stop_series(test_df[col], stop_to_idx)
        else:
            mapper = preprocess.get(col + "_mapping")
            if mapper is None:
                raise ValueError(f"Missing mapping for column '{col}' in preprocessing config.")
            test_df[col] = test_df[col].astype(str).map(mapper).fillna(-1).astype(int)
    
    X_test = test_df[features]
    y_test = test_df[target]

    print("\n===== TEST RESULTS =====")
    for name, model in models.items():
        evaluate(model, X_test, y_test, name)

    
if __name__ == "__main__":
    main()
