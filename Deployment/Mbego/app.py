import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained pipeline
with open("ModelPickle/diabetes_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Define selected features (must match training)
selected_features = [
    "encounter_id", "patient_nbr", "number_inpatient", "num_lab_procedures",
    "diag_1", "diag_2", "num_medications", "diag_3", "discharge_disposition_id",
    "time_in_hospital", "age", "number_diagnoses"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Collect user input
    user_input = {feature: float(request.form.get(feature, 0)) for feature in selected_features}

    # Convert input to DataFrame (to match pipeline format)
    input_df = pd.DataFrame([user_input])

    # Make prediction
    prediction = pipeline.predict(input_df)

    return render_template("index.html", prediction=f"Readmission Prediction: {prediction[0]}")

if __name__ == "__main__":
    app.run(debug=True)
