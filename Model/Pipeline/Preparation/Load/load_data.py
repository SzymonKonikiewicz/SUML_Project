import kagglehub
import os

import pandas as pd

def load_housing_data(
    path: str = "yasserh/housing-prices-dataset",
    sep: str = ",",
)-> pd.DataFrame:
    print("Path to dataset files:", kagglehub.dataset_download(path))
    file_path = os.path.join(kagglehub.dataset_download(path), "Housing.csv")
    return pd.read_csv(file_path, sep=sep)

def save_housing_data(
        df: pd.DataFrame,
        path: str,
        sep: str
)-> None:
    df.to_csv(path, sep=sep)