import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
from sklearn.preprocessing import StandardScaler

# Set background image
def set_background_image_local(image_path):
    with open(image_path, "rb") as file:
        data = file.read()
    base64_image = base64.b64encode(data).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/avif;base64,{base64_image}");
            background-size: contain;
            background-position: fit;
            background-repeat: repeat;
            background-attachment: fixed;
        }}     
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_image_local("BGIMG.avif")

# Load resources
@st.cache_resource
def load_lstm_model():
    return load_model('trained_model.h5', custom_objects={"LeakyReLU": LeakyReLU()})


@st.cache_resource
def load_scaler():
    with open('scaler.pkl', 'rb') as file:
        return pickle.load(file)


@st.cache_resource
def load_label_encoders():
    label_encoders = {}
    categorical_columns = ['tool_condition', 'machining_finalized', 'passed_visual_inspection']
    for column in categorical_columns:
        with open(f'label_encoder_{column}.pkl', 'rb') as file:
            label_encoders[column] = pickle.load(file)
    return label_encoders

# Initialize resources
model = load_lstm_model()
scaler = load_scaler()
label_encoders = load_label_encoders()
expected_features = scaler.feature_names_in_

# Preprocessing function
def preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    missing_cols = [col for col in expected_features if col not in df.columns]
    for col in missing_cols:
        default_value = st.number_input(f"Enter value for missing column '{col}':", value=0.0)
        df[col] = default_value

    df = df[expected_features]

    categorical_columns = ['tool_condition', 'machining_finalized', 'passed_visual_inspection']
    numeric_columns = [col for col in expected_features if col not in categorical_columns]

    for col in categorical_columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna("Unknown")
            df[col] = label_encoders[col].transform(df[col])

    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

    X_scaled = scaler.transform(df[numeric_columns])
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    return X_reshaped, df

# Prediction function
def make_predictions(X):
    predictions = model.predict(X)
    decoded_predictions = {}
    
    for i, column in enumerate(label_encoders.keys()):
        predicted_labels = np.argmax(predictions[i], axis=1)
        decoded_predictions[column] = label_encoders[column].inverse_transform(predicted_labels)
    
    return pd.DataFrame(decoded_predictions)

# Streamlit UI with Tabs
st.title("🔧 CNC Milling Tool Wear Prediction App")

tab1, tab2 = st.tabs(["📌 Overview", "🛠 Tool Wear Prediction"])

with tab1:
    st.markdown(
        """
        ### **📌 Introduction to CNC (Computer Numerical Control)**
        CNC (Computer Numerical Control) is an advanced manufacturing technology that enables automated control of machining tools through pre-programmed software and commands.  
        It is widely used in various industries for precision manufacturing, including automotive, aerospace, electronics, and medical sectors.  

        CNC machines replace traditional manual machining methods, significantly improving accuracy, efficiency, and productivity.  
        These machines are capable of performing highly complex cutting, drilling, milling, and shaping tasks with minimal human intervention.

        ### **⚙️ Key Benefits of CNC Machining**
        **1. High Precision and Accuracy**  
        CNC machines operate with extreme precision, ensuring consistent product quality with minimal errors.  

        **2. Automation and Efficiency**  
        Reduces manual labor, improves efficiency, and lowers production costs. CNC machines can run continuously, increasing productivity.  

        **3. Complex Design Capability**  
        CNC machines can manufacture intricate designs that would be difficult or impossible to achieve manually.  

        **4. Consistency and Repeatability**  
        Every part produced is identical, making CNC ideal for mass production industries.  

        **5. Faster Production and Time Savings**  
        CNC operations are much faster than manual machining, reducing lead times.  

        **6. Safety and Reduced Human Errors**  
        CNC machines operate autonomously, reducing direct human interaction with tools and lowering workplace accidents.  

        **7. Cost Savings in the Long Run**  
        While initial investment is high, CNC reduces material waste and labor costs over time.  

        **8. Integration with Modern Technologies**  
        AI, Machine Learning, and IoT integration enable predictive maintenance, fault detection, and real-time process optimization.  

        ### **🔩 General Applications of CNC Machining**
        CNC technology is widely used across various industries:  

        **✅ Automotive Industry** – Manufacturing engine parts, gearboxes, and chassis components.  
        **✈️ Aerospace Industry** – High-precision lightweight turbine blades and aircraft frames.  
        **📡 Electronics Industry** – Circuit boards, connectors, and semiconductor components.  
        **🏥 Medical Industry** – Surgical instruments, orthopedic implants, and prosthetics.  
        **🛡️ Defense & Military** – Precision parts for military vehicles, weapons, and aircraft.  
        **🏗️ Construction & Heavy Machinery** – Structural components for bridges, tunnels, and machinery.  
        **🎨 Consumer Goods** – Custom furniture, jewelry, musical instruments, and CNC engraving.  
        """
    )
    st.write("Upload your CSV file in the **Tool Wear Prediction** tab to make predictions.")

with tab2:
    st.header("🛠 Tool Wear Prediction")

    st.markdown(
        """
        ### **📌 Project Overview**
        This project analyzes **CNC milling machine performance** and detects faults using **deep learning techniques**.  
        The primary objectives of this project are:  
        - 🛠 **Tool Condition Prediction** (Unworn/Worn)  
        - ⚙️ **Machining Finalization Prediction** (Yes/No)  
        - 🔍 **Passed Visual Inspection Prediction** (Yes/No)  

        This is achieved using **LSTM-based deep learning models**, processing **sensor data** collected from CNC milling experiments.  
        The web app is built with **Streamlit** for an interactive experience.
        """
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        X, original_data = preprocess_data(uploaded_file)
        predictions_df = make_predictions(X)

        st.write("### **Predictions**")
        st.dataframe(predictions_df)

        final_results = pd.concat([original_data, predictions_df], axis=1)
        st.write("### **Full Data with Predictions**")
        st.dataframe(final_results)

        csv = final_results.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

