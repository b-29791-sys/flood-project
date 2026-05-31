from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final = np.array([features])
    prediction = model.predict(final)[0]

    return render_template("index.html", prediction_text=f"Flood Probability: {prediction}")

if __name__ == "__main__":
    app.run()
