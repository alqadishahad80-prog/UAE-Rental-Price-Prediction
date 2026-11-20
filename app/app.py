"""UAE Rental Price Prediction — Streamlit application.

GitHub location: app/app.py
Requires three .joblib files in models/ and UAE_properties.csv in data/.
This app recreates numerical, one-hot and engineered inputs from the notebooks.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
DATA_FILE = ROOT / "data" / "UAE_properties.csv"

st.set_page_config(page_title="UAE Rental Property Explorer", page_icon="🏠", layout="centered")
st.title("🏠 UAE Rental Property Explorer")
st.write("Estimate yearly rental prices, classify a rental category, and explore similar properties.")
st.info("Educational demonstration only — these models are not approved for real pricing or financial decisions.")


@st.cache_data
def get_property_choices():
    """Read only columns needed for the selection menus and city coordinates."""
    df = pd.read_csv(
        DATA_FILE,
        usecols=["City", "Type", "Furnishing", "Latitude", "Longitude"],
    )
    cities = sorted(df["City"].dropna().astype(str).unique().tolist())
    types = sorted(df["Type"].dropna().astype(str).unique().tolist())
    furnishing_options = sorted(df["Furnishing"].dropna().astype(str).unique().tolist())
    city_coordinates = df.groupby("City")[["Latitude", "Longitude"]].median()
    return cities, types, furnishing_options, city_coordinates


@st.cache_resource
def load_models():
    """Load the models already trained in notebooks 04, 05, and 06."""
    return {
        "regression": joblib.load(MODEL_DIR / "regression_model.joblib"),
        "classification": joblib.load(MODEL_DIR / "classification_model.joblib"),
        "clustering": joblib.load(MODEL_DIR / "clustering_model.joblib"),
    }


def make_property_features(city, property_type, furnishing, beds, baths, area, age, lat, lon):
    """Recreate the property features from notebooks 01 and 03."""
    bedroom_denominator = max(beds, 1)
    values = {
        "Beds": float(beds),
        "Baths": float(baths),
        "Area_in_sqft": float(area),
        "Age_of_listing_in_days": float(age),
        "Latitude": float(lat),
        "Longitude": float(lon),
        "Sqft_per_bedroom": float(area / bedroom_denominator),
        "Bathrooms_per_bedroom": float(baths / bedroom_denominator),
        "Total_rooms": float(beds + baths),
        "Is_studio": float(beds == 0),
        f"City = {city}": 1.0,
        f"Type = {property_type}": 1.0,
        f"Furnishing = {furnishing}": 1.0,
    }
    return values


def supervised_input(model_record, values):
    """Use exactly the training column names and order saved with the model."""
    columns = model_record["feature_columns"]
    return pd.DataFrame([{column: values.get(column, 0.0) for column in columns}], columns=columns)


def clustering_input(model_record, values):
    """Recreate log transforms and numeric scaling from notebook 06."""
    numeric_columns = model_record["numeric_columns"]
    dummy_columns = model_record["dummy_columns"]
    all_columns = numeric_columns + dummy_columns
    frame = pd.DataFrame([{col: values.get(col, 0.0) for col in all_columns}], columns=all_columns)
    for col in model_record["log_columns"]:
        frame[col] = np.log1p(frame[col].clip(lower=0))
    frame[numeric_columns] = model_record["scaler"].transform(frame[numeric_columns])
    return frame


required_files = [
    DATA_FILE,
    MODEL_DIR / "regression_model.joblib",
    MODEL_DIR / "classification_model.joblib",
    MODEL_DIR / "clustering_model.joblib",
]
missing = [str(path.relative_to(ROOT)) for path in required_files if not path.is_file()]
if missing:
    st.error("Files not found in GitHub: " + ", ".join(missing))
    st.stop()

try:
    cities, property_types, furnishings, coordinates = get_property_choices()
    saved_models = load_models()
except Exception as error:
    st.error("Unable to load data or saved models. Check the model files and Python package versions.")
    st.exception(error)
    st.stop()

st.subheader("Enter property details")
with st.form("property_form"):
    city = st.selectbox("City", cities, index=cities.index("Dubai") if "Dubai" in cities else 0)
    property_type = st.selectbox(
        "Property type", property_types,
        index=property_types.index("Apartment") if "Apartment" in property_types else 0,
    )
    furnishing = st.selectbox("Furnishing", furnishings)
    col1, col2 = st.columns(2)
    with col1:
        beds = st.number_input("Bedrooms", min_value=0, max_value=12, value=2, step=1)
        area = st.number_input("Area (sqft)", min_value=74, max_value=210254, value=1300, step=50)
    with col2:
        baths = st.number_input("Bathrooms", min_value=1, max_value=11, value=2, step=1)
        age = st.number_input("Days since listing", min_value=0, max_value=2500, value=50, step=1)

    typical_lat = float(coordinates.loc[city, "Latitude"])
    typical_lon = float(coordinates.loc[city, "Longitude"])
    with st.expander("Optional: adjust geographic coordinates"):
        st.caption("City-median coordinates are the default, not a specific property address.")
        latitude = st.number_input("Latitude", value=typical_lat, format="%.5f")
        longitude = st.number_input("Longitude", value=typical_lon, format="%.5f")

    submitted = st.form_submit_button("Predict property results", type="primary")

if submitted:
    values = make_property_features(
        city, property_type, furnishing, beds, baths, area, age, latitude, longitude
    )
    try:
        regression = saved_models["regression"]
        classification = saved_models["classification"]
        clustering = saved_models["clustering"]

        predicted_rent = float(regression["model"].predict(supervised_input(regression, values))[0])
        predicted_category = str(
            classification["model"].predict(supervised_input(classification, values))[0]
        )
        cluster = int(clustering["model"].predict(clustering_input(clustering, values))[0])

        st.subheader("Model results")
        st.metric("Estimated yearly rent", f"AED {predicted_rent:,.0f}")
        st.metric("Estimated rent category", predicted_category)
        st.metric("Property group (cluster)", f"Cluster {cluster + 1}")
        st.caption("Each result comes from a separate trained model. Cluster numbers are group IDs, not rankings.")
        if predicted_rent <= 0:
            st.warning("This model produced a non-positive rent prediction. Do not use it for pricing.")
        if hasattr(classification["model"], "predict_proba"):
            probabilities = classification["model"].predict_proba(
                supervised_input(classification, values)
            )[0]
            classes = classification["model"].classes_
            st.write("Classification model scores:")
            st.dataframe(
                pd.DataFrame({"Category": classes, "Model score": probabilities}).sort_values(
                    "Model score", ascending=False
                ),
                hide_index=True,
            )
        st.warning("Model reliability must be checked on new listings before deployment for real decisions.")
    except Exception as error:
        st.error("Prediction failed. Verify that the uploaded model files match notebooks 04–06.")
        st.exception(error)

st.divider()
st.caption("Source: UAE Properties dataset. Models trained in the project's Python notebooks.")
