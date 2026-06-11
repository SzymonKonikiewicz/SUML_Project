import joblib
import pandas as pd
import json
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from Model.Models.linear_regressor import LinearRegressor
from Model.Models.random_forest_regressor import RandomForestRegressor


TARGET_COL = "price"

NUMERICAL_FEATURES = [
    "area",
]

CATEGORICAL_FEATURES = [
    "bedrooms",
    "bathrooms",
    "stories",
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "parking",
    "prefarea",
    "furnishingstatus",
]


def build_model_pipeline(regressor) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numerical", "passthrough", NUMERICAL_FEATURES),
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    return model_pipeline


def train_pipeline(
    df: pd.DataFrame,
    regressor,
    model_path: str = "Model/Artifacts/default_pipeline.joblib",
) -> Pipeline:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model_pipeline = build_model_pipeline(regressor=regressor)

    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.3f}")

    metrics = {
        "model_path": model_path,
        "mae": float(mae),
        "r2": float(r2),
    }

    model_path_obj = Path(model_path)
    model_path_obj.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, model_path_obj)

    metrics_path = model_path_obj.with_name(
        f"{model_path_obj.stem}_metrics.json"
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, model_path)

    return model_pipeline

def load_metrics(model_path: str) -> dict:
    model_path_obj = Path(model_path)

    metrics_path = model_path_obj.with_name(
        f"{model_path_obj.stem}_metrics.json"
    )

    if not metrics_path.exists():
        return {}

    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_pipeline(
    model_path: str = "Model/Artifacts/default_pipeline.joblib",
) -> Pipeline:
    return joblib.load(model_path)


def predict_price(
    input_data: dict,
    model_path: str = "Model/Artifacts/default_pipeline.joblib",
) -> float:
    model_pipeline = load_pipeline(model_path)

    input_df = pd.DataFrame([input_data])
    prediction = model_pipeline.predict(input_df)[0]

    metrics = load_metrics(model_path)

    return {
        "predicted_price": float(prediction),
        "model_metrics": metrics
    }


  