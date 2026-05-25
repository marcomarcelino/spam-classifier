# 🐳 Spam Classifier — Containerized ML API

A spam email/SMS classifier built from scratch and packaged as a Docker container, making it portable and deployment-ready across any environment.

## 📊 Model Performance

- **Algorithm:** Naive Bayes + TF-IDF Vectorization
- **Dataset:** UCI SMS Spam Collection (5,572 messages)
- **Accuracy:** 98%

## 🏗️ Architecture"# spam-classifier"

## 🚀 Run with Docker

```bash
docker pull marcomarcelino/spam-classifier
docker run -p 5000:5000 spam-classifier
```

## 📡 API Usage

**Predict spam:**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Congratulations! You won a FREE iPhone!"}'
```

**Response:**

```json
{
  "message": "Congratulations! You won a FREE iPhone!",
  "prediction": "spam"
}
```

## 🛠️ Tech Stack

- Python 3.11
- scikit-learn 1.6.1
- Flask 3.0.0
- Docker

## 💡 What I Learned

- Containerization solves the "it works on my machine" problem
- Version pinning is essential for reproducible ML environments
- DevOps principles are what bridge ML models from notebooks to production
