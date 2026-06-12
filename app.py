import streamlit as st
import pandas as pd
import joblib
import os
import json

#=============
#Loading the trained model

#Filepaths to models
model_path_linear = os.path.join("Model", "Artifacts", "linear_regressor_pipeline.joblib")
model_path_rforest = os.path.join("Model", "Artifacts", "random_forest_regressor_pipeline.joblib")

#Filepaths to metrics
metrics_path_linear = os.path.join("Model", "Artifacts", "linear_regressor_pipeline_metrics.json")
metrics_path_rforest = os.path.join("Model", "Artifacts", "random_forest_regressor_pipeline_metrics.json")

#Filepath to data, load data
data_path = os.path.join("Model", "Data", "Clean", "housing_calc_read.csv")
df = pd.read_csv(data_path)

#Loading style sheets
def load_css(style):
    try:
        with open(style) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Nie można znaleźć pliku CSS: {style}")

#Loading metrics
def load_metrics(metrics_path):
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Nie można znaleźć pliku z metrykami: {metrics_path}")
        return None

#Linear regression model
@st.cache_resource
def load_model_linear():
    return joblib.load(model_path_linear)

#Random forest model
@st.cache_resource
def load_model_rforest():
    return joblib.load(model_path_rforest)

#Loading models
linear_regression = load_model_linear()
random_forest = load_model_rforest()

#Available models and metrics:
available_models = {
    "Random Forest Regressor": random_forest, 
    "Linear Regression" : linear_regression
    # Add more
}

available_metrics = {
    "Random Forest Regressor": load_metrics(metrics_path_rforest),
    "Linear Regression": load_metrics(metrics_path_linear)
    # Add more
}

#=================


#Layout===========


st.set_page_config(
    page_title="SUML_Project",
    page_icon=":house:",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css("style.css")

st.title("Projekt SUML - Predykcja cen nieruchomości")
st.write("Nasza aplikacja umożliwia szacowanie wartosci nieruchomości luksusowych " \
"na podstawie danych dotyczących cen mieszkań w dzielnicach Nowego Jorku." \
" Wykorzystujemy dwa modele: Random Forest Regressor oraz Linear Regression, " \
"które zostały przeszkolone na danych dotyczących różnych cech mieszkań wyszczególnionych poniżej. " \
"Dzięki temu możemy dostarczyć użytkownikom dokładne prognozy cen nieruchomości " \
"na podstawie wprowadzonych parametrów.")

st.divider()

# New data frame without a price column
new_df = df.drop(columns=["price"])

column_names = new_df.columns.tolist()

#Fields shown to the user
available_fields = {
    column_names[0]: "Powierzchnia (m²)",
    column_names[1]: "Liczba sypialni",
    column_names[2]: "Liczba łazienek",
    column_names[3]: "Piętro",
    column_names[4]: "Przy głównej ulicy",
    column_names[5]: "Pokój gościnny",
    column_names[6]: "Piwnica",
    column_names[7]: "Centralne ogrzewanie",
    column_names[8]: "Klimatyzacja",
    column_names[9]: "Liczba miejsc parkingowych",
    column_names[10]: "Dobra lokalizacja",
    column_names[11]: "Status umeblowania"
    # Add more if needed or if dataset changes
}   

basic_translation = {
    "yes": "Tak",
    "no": "Nie",
    "furnished": "Umeblowane",
    "unfurnished": "Nieumeblowane",
    "semi-furnished": "Pół umeblowane"
}

# Side panel with model choice
with st.sidebar:
    st.header("Wybierz model do predykcji ceny: ")
    model_choice = st.selectbox(
        label="Model",
        options=list(available_models.keys()),
        index=0,
    )

# Panel for user input
user_data = {}

st.header("Parametry mieszkania")
with st.form("user_input_form", border=True):
    columns = st.columns(4)

    for index, c in enumerate(column_names):

        # Automatic column assignment for user input fields
        actual = columns[index % 4]

        unique_values = new_df[c].dropna().unique()

        # mean = float(new_df[c].mean())

        with actual:
            if new_df[c].dtype in ["object", "string"] or len(unique_values) <= 2:

                options = unique_values.tolist()

                user_data[c] = st.selectbox(
                    label=available_fields.get(c, c), 
                    options=options, 
                    index=None,
                    placeholder="Wybierz opcję",
                    format_func=lambda x: basic_translation.get(x, x)
                )
            else:
                user_data[c] = st.number_input(
                    label=f"{available_fields.get(c, c)}", 
                    value=None,
                    placeholder="Wprowadź wartość",
                    step=1,
                    min_value=0)

    st.divider()

    submitted = st.form_submit_button("Wygeneruj predykcję ceny!", type="primary", use_container_width=True)

# test button using forms

if submitted:
    # Check if user has filled all fields, if not show error message with missing fields
    missing_fields = [available_fields.get(key, key) for key, value in user_data.items() if value is None]

    if missing_fields:
        st.error(f"Proszę uzupełnić wszystkie pola przed wygenerowaniem predykcji. Brakujące pola: {', '.join(missing_fields)}")
        st.stop()


    input_df = pd.DataFrame([user_data])

    st.write("Podgląd inputu użytkownika")

    st.dataframe(input_df, hide_index=True)

    st.divider()

    try:
        model_used = available_models.get(model_choice)

        prediction = model_used.predict(input_df)[0]
        formatted_price = f"{prediction:,.0f}".replace(",", " ")

        st.metric(
            label=f"Szacowana wartość ({model_choice})",
            value=f"{formatted_price} USD",
        )

        mae_value = available_metrics[model_choice].get("mae", None)
        r2_value = available_metrics[model_choice].get("r2", None)

        st.divider()
        st.write("Metryki modelu:")
        st.metric(
            label="Wartość wskaźnika MAE",
            value=f"{mae_value:.2f}"
        )

        st.metric(
            label="Wartość wskaźnika R²",
            value=f"{r2_value:.4f}"
        )

        st.balloons()

    except Exception as e:
        st.error(f"Wystąpił błąd podczas predykcji: {e}")
