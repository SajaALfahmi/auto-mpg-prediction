# -----------------------------
# Imports
# -----------------------------
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="MPG Prediction (SVR)", page_icon="🚗", layout="centered")

# -----------------------------
# Load Model and Scaler
# -----------------------------
@st.cache_resource
def load_artifacts():
    with open("best_svr_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# -----------------------------
# Helpers
# -----------------------------
FEATURES_ORDER = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model year",
    "origin",
]

def load_and_clean_dataset(path="auto-mpg.csv"):
    df = pd.read_csv(path)

    # clean horsepower + drop missing
    df["horsepower"] = df["horsepower"].replace("?", np.nan).astype(float)
    df = df.dropna().copy()

    # drop car name if exists
    if "car name" in df.columns:
        df = df.drop(columns=["car name"])

    return df

def make_feature_matrix(df):
    # Ensure correct columns + correct order
    X = df[FEATURES_ORDER].copy()
    y = df["mpg"].copy()
    return X, y

def compute_metrics_from_test_csv(path="svr_predictions.csv"):
    # This is best because it's usually your TEST set predictions
    pred_df = pd.read_csv(path)
    y_true = pred_df["Actual MPG"].values
    y_pred = pred_df["Predicted MPG (SVR)"].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2, y_true, y_pred

# -----------------------------
# UI
# -----------------------------
st.title("🚗 MPG Prediction App")
st.caption("Predict fuel efficiency (MPG) using a trained SVR model + show metrics & visualization.")

# -----------------------------
# Metrics + Visualization
# -----------------------------
st.subheader("Model Performance")

use_test_csv = st.checkbox("Use saved TEST predictions (svr_predictions.csv) for metrics/plot (recommended)", value=True)

rmse = r2 = None
y_true = y_pred = None

if use_test_csv:
    try:
        rmse, r2, y_true, y_pred = compute_metrics_from_test_csv("svr_predictions.csv")
        st.success("Loaded metrics from svr_predictions.csv (test set).")
    except Exception as e:
        st.warning(f"Couldn't load svr_predictions.csv. Falling back to evaluating on full dataset. ({e})")
        use_test_csv = False

if not use_test_csv:
    try:
        df = load_and_clean_dataset("auto-mpg.csv")
        X, y = make_feature_matrix(df)
        X_scaled = scaler.transform(X)  # scaler trained earlier in your notebook
        preds = model.predict(X_scaled)

        y_true = y.values
        y_pred = preds

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        st.info("Note: These metrics are computed on the full dataset (includes training data).")
    except Exception as e:
        st.error(f"Dataset evaluation failed. Make sure auto-mpg.csv is in the same folder. Error: {e}")

if rmse is not None and r2 is not None:
    c1, c2 = st.columns(2)
    c1.metric("RMSE", f"{rmse:.2f}")
    c2.metric("R²", f"{r2:.2f}")

    st.subheader("Actual vs Predicted (Scatter)")
    fig = plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual MPG")
    plt.ylabel("Predicted MPG")
    plt.title("Actual vs Predicted MPG")
    st.pyplot(fig)

# -----------------------------
# User Inputs (MUST match model features)
# -----------------------------
st.subheader("Predict MPG for a New Car")

cylinders = st.number_input("Cylinders", min_value=3, max_value=12, value=4, step=1)
displacement = st.number_input("Displacement", min_value=50.0, max_value=500.0, value=150.0, step=1.0)
horsepower = st.number_input("Horsepower", min_value=40.0, max_value=250.0, value=100.0, step=1.0)
weight = st.number_input("Weight", min_value=1500, max_value=6000, value=3000, step=10)
acceleration = st.number_input("Acceleration", min_value=5.0, max_value=25.0, value=15.0, step=0.1)
model_year = st.number_input("Model Year", min_value=70, max_value=90, value=76, step=1)

origin_label = st.selectbox("Origin", ["USA (1)", "Europe (2)", "Japan (3)"])
origin = 1 if origin_label.startswith("USA") else (2 if origin_label.startswith("Europe") else 3)

if st.button("Predict MPG"):
    input_row = np.array([[cylinders, displacement, horsepower, weight, acceleration, model_year, origin]], dtype=float)
    input_scaled = scaler.transform(input_row)
    pred = model.predict(input_scaled)[0]
    st.success(f"✅ Estimated MPG: {pred:.2f}")
