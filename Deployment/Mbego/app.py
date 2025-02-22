import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the pickled model
#model_path = r"N:\Moringa\afterM\joseline 001\Diabetes_130-US_Hospitals_1999-2008\Deployment\Mbego\ModelPickle\best_random_forest_model.pkl.bz2" 
#model = pickle.load(open(model_path, "rb"))

import bz2

model_path = r"N:\Moringa\afterM\joseline 001\Diabetes_130-US_Hospitals_1999-2008\Deployment\Mbego\ModelPickle\best_random_forest_model.pkl.bz2"

# Open the compressed file correctly
with bz2.BZ2File(model_path, "rb") as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    data = [float(x) for x in request.form.values()]
    final_data = pd.DataFrame([data])
    prediction = model.predict(final_data)
    return render_template('index.html', prediction_text=f'Prediction: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)
