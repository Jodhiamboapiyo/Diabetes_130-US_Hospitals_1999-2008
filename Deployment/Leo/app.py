from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and encoder
with open("diabetes_model.pkl", "rb") as file:
    loaded_objects = pickle.load(file)

model = loaded_objects["model"]
encoder = loaded_objects["encoder"]

# Define numerical and categorical columns
numerical_columns = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses"
]

categorical_columns = [
    "race", "gender", "age", "max_glu_serum", "A1Cresult", "metformin",
    "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone",
    "acarbose", "miglitol", "tolazamide", "insulin", "glyburide-metformin",
    "change", "diabetesMed", "diag_1_category", "diag_2_category",
    "diag_3_category"
]

# Ensure final column order matches training data
final_column_order = numerical_columns + list(encoder.get_feature_names_out(categorical_columns))

# Mapping to fix age format issues
age_mapping = {
    "0-10": "[0-10)", "10-20": "[10-20)", "20-30": "[20-30)", 
    "30-40": "[30-40)", "40-50": "[40-50)", "50-60": "[50-60)", 
    "60-70": "[60-70)", "70-80": "[70-80)", "80-90": "[80-90)", 
    "90-100": "[90-100)"
}

@app.route("/")
def home():
    return render_template("homepage.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input
        input_data = {col: request.form[col] for col in numerical_columns + categorical_columns}
        input_df = pd.DataFrame([input_data])

        # Convert numerical features to correct type
        input_df[numerical_columns] = input_df[numerical_columns].apply(pd.to_numeric, errors="coerce")

        # Fix age format before encoding
        input_df["age"] = input_df["age"].map(age_mapping).fillna(input_df["age"])  

        # Convert categorical columns to strings (fix unknown category issue)
        input_df[categorical_columns] = input_df[categorical_columns].astype(str)

        # Apply OneHotEncoder to categorical columns
        encoded_data = encoder.transform(input_df[categorical_columns])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))

        # Merge numerical and encoded categorical features
        processed_input = pd.concat([input_df[numerical_columns], encoded_df], axis=1)

        # Ensure the column order matches training data
        processed_input = processed_input.reindex(columns=final_column_order, fill_value=0)

        # Make prediction
        prediction = model.predict(processed_input)

        # Interpret the result
        result = "Not Readmitted" if prediction[0] == 0 else "Readmitted"

        return render_template("homepage.html", prediction=result)

    except Exception as e:
        return render_template("homepage.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
