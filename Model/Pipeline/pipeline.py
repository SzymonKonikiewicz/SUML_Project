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


def build_model_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numerical", "passthrough", NUMERICAL_FEATURES),
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegressor()),
        ]
    )

    return model_pipeline


def train_pipeline(
    df: pd.DataFrame,
    model_path: str = "Model/Artifacts/housing_price_pipeline.joblib",
) -> Pipeline:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model_pipeline = build_model_pipeline()

    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.3f}")

    metrics = {
        "mae": float(mae),
        "r2": float(r2),
    }

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, model_path)

    metrics_path = Path(model_path).with_name("metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, model_path)

    return model_pipeline


def load_pipeline(
    model_path: str = "Model/Artifacts/housing_price_pipeline.joblib",
) -> Pipeline:
    return joblib.load(model_path)


def predict_price(
    input_data: dict,
    model_path: str = "Model/Artifacts/housing_price_pipeline.joblib",
) -> float:
    model_pipeline = load_pipeline(model_path)

    input_df = pd.DataFrame([input_data])
    prediction = model_pipeline.predict(input_df)[0]

    return float(prediction)