from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

iris = load_iris()
model = RandomForestClassifier(random_state=42)
model.fit(iris.data, iris.target)

@app.route("/")
def home():
    return jsonify({
        "message": "Iris ML Prediction API is running",
        "status": "success"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features", [])
    if len(features) != 4:
        return jsonify({"error": "Provide exactly 4 features"}), 400

    prediction = model.predict([features])[0]
    return jsonify({
        "prediction": iris.target_names[prediction].item()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
