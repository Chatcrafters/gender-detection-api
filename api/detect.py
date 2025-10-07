from flask import Flask, request, jsonify
from transformers import pipeline

# 🧠 Kleineres Modell → läuft stabil auf Vercel Free
classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-1"
)

app = Flask(__name__)

@app.route("/api/detect", methods=["POST"])
def detect_gender():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"gender": "neutral"}), 200

    # Nur ersten Teil analysieren für Performance
    sample = text[:300]

    labels = ["male", "female", "neutral"]
    result = classifier(sample, candidate_labels=labels)

    gender = result["labels"][0]
    return jsonify({"gender": gender}), 200

@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "ok", "message": "Gender API running"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
