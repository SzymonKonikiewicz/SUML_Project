from Model.Pipeline.Preparation.Load.load_data import (
    load_housing_data,
    save_housing_data,
)
from Model.Pipeline.Preparation.Clean.clean_data import (
    describe_column,
    establish_col_type,
    clean_dataframe,
)

from Model.Models.linear_regressor import LinearRegressor
from Model.Pipeline.pipeline import train_pipeline

df = load_housing_data()
save_housing_data(df, "Model/Data/Raw/housing_raw.csv", ",")
print(df.head())

print(f"Columns: {df.columns.to_list()}")
print(f"Shape: {df.shape}")

column_types = {}
for col in df.columns:
    print(f"Column {col} has type: {establish_col_type(df[col])}")
    column_types[col] = establish_col_type(df[col])

for col in df.columns:
    describe_column(df, col, establish_col_type(df[col]))

clean_dataframe(
    df=df,
    column_types=column_types,
    target_col="price",
    random_state=42,
    save_path="Model/Data/Clean/housing_calc_read.csv",
)

train_pipeline(
    df=df,
    regressor=LinearRegressor(),
    model_path="Model/Artifacts/linear_regressor_pipeline.joblib",
)
