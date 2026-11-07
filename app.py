import streamlit as st
import pandas as pd
import joblib

# Load the saved model and preprocessing objects
model = joblib.load("logistic_regression_heart.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# These are the five numerical columns used when scaler.pkl was fitted
numeric_columns = list(scaler.feature_names_in_)

# Basic consistency checks to catch mismatched saved files early
if list(model.feature_names_in_) != list(columns):
    st.error("The model and columns.pkl do not match. Please check your saved training files.")
    st.stop()

if not set(numeric_columns).issubset(columns):
    st.error("The scaler's numerical columns are not present in columns.pkl.")
    st.stop()

st.title("❤️ Heart Disease Prediction")
st.write("Enter the required details to get a model prediction.")
st.divider()
st.subheader("Patient Details")

Age = st.slider("Age", min_value=18, max_value=100, value=30)
Sex = st.selectbox("Sex", options=["M", "F"])
Chest_Pain = st.selectbox("Chest Pain Type", options=["ATA", "NAP", "TA", "ASY"])
Resting_BP = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
Cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
Fasting_BS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
Resting_ECG = st.selectbox("Resting Electrocardiographic Results", options=["Normal", "ST", "LVH"])
Max_HR = st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
Exercise_Angina = st.selectbox("Exercise Induced Angina", options=["Y", "N"])
Oldpeak = st.slider("Oldpeak (ST depression)", min_value=0.0, max_value=6.0, value=1.0)
ST_Slope = st.selectbox("Slope of the peak exercise ST segment", options=["Up", "Flat", "Down"])

if st.button("Predict"):
    # Start with the six numerical/binary input columns.
    # FastingBS is intentionally NOT scaled because scaler.pkl was fitted on only five columns.
    raw_input = {
        "Age": Age,
        "RestingBP": Resting_BP,
        "Cholesterol": Cholesterol,
        "FastingBS": Fasting_BS,
        "MaxHR": Max_HR,
        "Oldpeak": Oldpeak,
    }

    # Add one-hot encoded categorical values.
    # All categories are represented by the fixed columns list; the selected category gets 1.
    for col in columns:
        if col.startswith("Sex_"):
            raw_input[col] = int(col == f"Sex_{Sex}")
        elif col.startswith("ChestPainType_"):
            raw_input[col] = int(col == f"ChestPainType_{Chest_Pain}")
        elif col.startswith("RestingECG_"):
            raw_input[col] = int(col == f"RestingECG_{Resting_ECG}")
        elif col.startswith("ExerciseAngina_"):
            raw_input[col] = int(col == f"ExerciseAngina_{Exercise_Angina}")
        elif col.startswith("ST_Slope_"):
            raw_input[col] = int(col == f"ST_Slope_{ST_Slope}")

    # Build exactly the 20 columns in the same order used to train the model.
    input_data = pd.DataFrame([raw_input]).reindex(columns=columns, fill_value=0).astype(float)

    # Scale ONLY the five columns used to fit scaler.pkl.
    input_data.loc[:, numeric_columns] = scaler.transform(input_data[numeric_columns])

    # Final safety check: LogisticRegression must receive all 20 features.
    if input_data.shape[1] != model.n_features_in_:
        st.error(
            f"Feature mismatch: input has {input_data.shape[1]} features, "
            f"but model expects {model.n_features_in_}."
        )
        st.stop()

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("The model predicts that the patient is likely to have heart disease.")
    else:
        st.success("The model predicts that the patient is unlikely to have heart disease.")

    st.caption("This is a machine-learning prediction, not a medical diagnosis. Please consult a qualified healthcare professional.")
