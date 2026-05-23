import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    message = data.get('message', '')
    vec = vectorizer.transform([message])
    pred = model.predict(vec)[0]
    result = 'spam' if pred == 1 else 'ham'
    return jsonify({'message': message, 'prediction': result})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)