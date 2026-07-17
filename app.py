from flask import Flask, request, jsonify, render_template
import pickle
import re
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

# Manual stopwords
stop_words = set([
    "i","me","my","myself","we","our","you","your","he","she","it",
    "they","is","are","was","were","the","a","an","and","or","to","of","in"
])

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['review']

    cleaned = clean_text(data)

    vector = tfidf.transform([cleaned]).toarray()

    result = model.predict(vector)[0]

    return jsonify({'sentiment': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)