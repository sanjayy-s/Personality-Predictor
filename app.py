from flask import Flask, request, jsonify
import joblib
import re
from flask_cors import CORS  # Allows frontend JS to talk to Flask

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Load model and TF-IDF vectorizer
model = joblib.load('personality_model.pkl')
tfidf = joblib.load('vectorizer.pkl')

def clean_post(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_text = data.get('text', '')

    # Clean and vectorize input
    cleaned = clean_post(user_text)
    vectorized = tfidf.transform([cleaned])

    # Predict using the trained model
    prediction = model.predict(vectorized)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
