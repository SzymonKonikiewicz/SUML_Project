import streamlit as st
import pandas as pd
import os

from Model.Pipeline.pipeline import predict_price

# contains paths to trained models
available_models = {
    "linear_regressor": "Model/Artifacts/linear_regressor_pipeline.joblib",
    "random_forest_regressor": "Model/Artifacts/random_forest_regressor_pipeline.joblib"
}

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

#
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

    model_used = "random_forest_regressor" # here just replace with user choice

    if not available_models.__contains__(model_used):
        raise NameError(f"Provide existing model, choose one from: {[key for key in available_models.keys()]}")

    prediction = predict_price(
        input_data=user_data,
        model_path=available_models[model_used] 
    )
    formatted_price = f"{prediction:,.0f}".replace(",", " ")

    st.metric(
        label=f"Predicted house price with '{model_used}' algorithm",
        value=f"{formatted_price} USD",
    )
