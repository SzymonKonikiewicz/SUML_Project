# 🗽 NYC Real Estate Price Predictor (SUML Project)

A Streamlit web application designed to estimate the value of luxury real estate in New York City[cite: 2]. The application allows users to input various property characteristics and generates an estimated price using trained machine learning models[cite: 2].

## 🛠 Technologies Used
* **Languages:** Python 3.13
* **Machine Learning:** Scikit-learn (Random Forest, Linear Regression)
* **Framework:** Streamlit
* **Infrastructure:** Docker, Terraform

## 🧠 Machine Learning Models
Users can choose between two models for prediction from the sidebar menu[cite: 2]:
* **Random Forest Regressor**[cite: 2]
* **Linear Regression**[cite: 2]

## 📋 Input Parameters
The application calculates predictions based on the following property features[cite: 2]:
* **Basic Specs:** Area (m²), Number of bedrooms, Number of bathrooms, Floor[cite: 2]
* **Amenities:** Guest room, Basement, Central heating, Air conditioning, Number of parking spaces[cite: 2]
* **Location & Status:** Proximity to main road, Good location, Furnishing status[cite: 2]

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
# Build the image
docker build -t suml_project .

# Run the container
docker run -p 8501:8501 suml_project
```
### Option 2: Local Python Environment (Conda)
```bash
# Create and activate the environment
conda create -n suml_project python=3.13
conda activate suml_project

# Install dependencies
python -m pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```
