from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@app.route("/api/detect", methods=["POST"])
def detect_gender():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"gender": "neutral"}), 200

    sample = text[:300]
    labels = ["male", "female", "neutral"]
    result = classifier(sample, candidate_labels=labels)
    return jsonify({"gender": result["labels"][0]}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)

