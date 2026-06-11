import streamlit as st
import pandas as pd
import joblib
import os

from Model.Pipeline.pipeline import predict_price

#Loading the trained model

#Filepaths to models
model_path_linear = os.path.join("Model", "Artifacts", "linear_regressor_pipeline.joblib")
model_path_rforest = os.path.join("Model", "Artifacts", "random_forest_regressor_pipeline.joblib")

#First model: linear regression
@st.cache_resource
def load_model_linear():
    return
joblib.load(model_path_linear)

#Second model: Random forest
@st.cache_resource
def load_model_rforest():
    return
joblib.load(model_path_rforest)


linear_regression = load_model_linear()
random_forest = load_model_rforest()

st.markdown(
    """
<style>
    /* Celujemy w wewnętrzne identyfikatory przycisków Streamlit i je ukrywamy */
    [data-testid="stNumberInputStepUp"] {display: none;}
    [data-testid="stNumberInputStepDown"] {display: none;}
</style>
""",
    unsafe_allow_html=True,
)


#
data_path = os.path.join("Model", "Data", "Clean", "housing_calc_read.csv")
df = pd.read_csv(data_path)


# Title of the app
st.title("SUML_Project")
st.write("Tutaj będzie krótki opis naszego projektu")

st.divider()

st.header("Proszę wpisać dane: ")

user_data = {}

# New data frame without a price column
new_df = df.drop(columns=["price"])

column_names = new_df.columns.tolist()


columns = st.columns(4)


for index, c in enumerate(column_names):
    actual = columns[index % 4]

    unique_values = new_df[c].dropna().unique()

    # mean = float(new_df[c].mean())

    with actual:
        if new_df[c].dtype in ["object", "string"] or len(unique_values) <= 2:
            options = unique_values.tolist()

            user_data[c] = st.selectbox(label=f"{c}", options=options)
        else:
            user_data[c] = st.number_input(label=f"{c}", value=0.0, step=None)

st.divider()

# test button

if st.button("Przewidywanie ceny", type="primary", use_container_width=True):
    input_df = pd.DataFrame([user_data])

    st.write("Podgląd inputu użytkownika")

    st.dataframe(input_df, hide_index=True)

    st.divider()

    #st.subheader(f"Predicted price: ")

    prediction = predict_price(input_data=user_data)
    formatted_price = f"{prediction:,.0f}".replace(",", " ")

    st.metric(
        label="Predicted house price",
        value=f"{formatted_price} USD",
    )
