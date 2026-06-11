import pandas as pd
import numpy as np
from enum import Enum
from pathlib import Path


class ColType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    NOMINAL = "nominal"


def establish_col_type(col=pd.Series) -> type:
    if pd.api.types.is_numeric_dtype(col):
        unique_count = col.nunique(dropna=True)

        if unique_count <= 10:
            return ColType.CATEGORICAL

        return ColType.NUMERICAL

    if pd.api.types.is_bool_dtype(col):
        return ColType.CATEGORICAL

    if pd.api.types.is_object_dtype(col) or pd.api.types.is_categorical_dtype(col):
        unique_count = col.nunique(dropna=True)

        if unique_count >= 20:
            return ColType.NOMINAL

        return ColType.CATEGORICAL

    return ColType.CATEGORICAL


def describe_column(
    df: pd.DataFrame,
    col: str,
    col_type: ColType,
) -> None:
    series = df[col]

    row_count = len(series)
    missing_count = series.isna().sum()
    missing_pct = missing_count / row_count * 100
    unique_count = series.nunique(dropna=True)
    unique_pct = unique_count / row_count * 100

    print("=" * 60)
    print(f"Column: {col}")
    print(f"Logical type: {col_type.value}")
    print(f"Pandas dtype: {series.dtype}")
    print(f"Rows: {row_count}")
    print(f"Missing values: {missing_count} ({missing_pct:.2f}%)")
    print(f"Unique values: {unique_count} ({unique_pct:.2f}%)")

    if col_type == ColType.NUMERICAL:
        print("\nNumerical summary:")
        print(f"Min: {series.min()}")
        print(f"Max: {series.max()}")
        print(f"Mean: {series.mean():.2f}")
        print(f"Median: {series.median():.2f}")
        print(f"Std: {series.std():.2f}")

        print("\nQuantiles:")
        print(series.quantile([0.25, 0.5, 0.75]))

        zero_count = (series == 0).sum()
        print(f"\nZero values: {zero_count} ({zero_count / row_count * 100:.2f}%)")

    elif col_type == ColType.CATEGORICAL:
        print("\nCategorical summary:")
        print(f"Is binary: {unique_count == 2}")

        print("\nValue counts:")
        print(series.value_counts(dropna=False))

        print("\nValue percentages:")
        print((series.value_counts(dropna=False, normalize=True) * 100).round(2))

    elif col_type == ColType.NOMINAL:
        print("\nNominal / label summary:")

        print("\nExample values:")
        print(series.dropna().astype(str).head(10).to_list())

        print("\nMost common values:")
        print(series.value_counts(dropna=False).head(10))

        if unique_pct > 400:
            print("\nWarning: This column looks like an identifier or free-text label.")
            print("It may not be useful directly as a model feature.")


# if save_path is None then no csv will be created nor overrided
def clean_column(
    df: pd.DataFrame,
    col: str,
    col_type: ColType,
    random_state: int = 42,
    save_path: str | None = None,
) -> pd.DataFrame:
    cleaned_df = df.copy()
    rng = np.random.default_rng(random_state)

    missing_mask = cleaned_df[col].isna()
    missing_count = missing_mask.sum()

    if missing_count == 0:
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cleaned_df.to_csv(save_path, index=False)
        return cleaned_df

    if col_type == ColType.NUMERICAL:
        cleaned_df = cleaned_df.dropna(subset=[col])

    elif col_type == ColType.CATEGORICAL:
        value_distribution = cleaned_df[col].dropna().value_counts(normalize=True)

        sampled_values = rng.choice(
            value_distribution.index.to_numpy(),
            size=missing_count,
            p=value_distribution.to_numpy(),
        )

        cleaned_df.loc[missing_mask, col] = sampled_values

    elif col_type == ColType.NOMINAL:
        cleaned_df[col] = cleaned_df[col].fillna("unknown")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(save_path, index=False)

    return cleaned_df


def clean_dataframe(
    df: pd.DataFrame,
    column_types: dict[str, ColType],
    target_col: str | None = None,
    random_state: int = 42,
    save_path: str | None = None,
) -> pd.DataFrame:
    cleaned_df = df.copy()

    # Jeśli target ma braki, zawsze usuwamy rekordy
    if target_col is not None:
        cleaned_df = cleaned_df.dropna(subset=[target_col])

    for col, col_type in column_types.items():
        if col == target_col:
            continue

        cleaned_df = clean_column(
            df=cleaned_df,
            col=col,
            col_type=col_type,
            random_state=random_state,
            save_path=None,  # nie zapisujemy po każdej kolumnie
        )

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(save_path, index=False)

    return cleaned_df
