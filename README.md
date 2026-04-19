# ArXiv Paper Classification System (DistilBERT + ML Baselines)

An end-to-end NLP system to classify research paper abstracts into categories such as **cs.AI, cs.NE, cs.CV, cs.SY, cs.DS**, comparing classical ML approaches with a fine-tuned Transformer model and deploying the best model via FastAPI and Docker.

---

## Overview

This project explores the effectiveness of **classical machine learning vs transformer-based models** for text classification on research abstracts.

It includes:
- TF-IDF + Logistic Regression
- TF-IDF + Support Vector Machine
- Fine-tuned DistilBERT (HuggingFace)

The best-performing model is deployed as a **real-time inference API**.

---

## Results

| Model                     | Accuracy |
|--------------------------|----------|
| Logistic Regression      | 0.82     |
| SVM                      | 0.81     |
| DistilBERT (fine-tuned)  | **0.86** |

### Key Insight
Transformer-based models significantly outperform classical approaches, especially in **semantically overlapping classes**.

Example improvement:
- Class 2 F1-score: **0.70 → 0.93**

---

## Problem Statement

Given a research paper abstract, predict its category:
- `cs.AI` – Artificial Intelligence  
- `cs.NE` – Neural & Evolutionary Computing  
- `cs.CV` – Computer Vision  
- `cs.SY` – Systems  
- `cs.DS` – Data Structures  

---

## Architecture
Text Input → Tokenizer → DistilBERT → Softmax → Label Mapping → API Response


---

## Features

- 🔍 Comparative analysis (TF-IDF vs Transformer)
- 🤖 Fine-tuned DistilBERT model
- 📊 Class-wise evaluation (precision, recall, F1)
- ⚡ FastAPI-based inference service
- 🐳 Dockerized deployment
- 🎯 Confidence-aware predictions

---

## Project Structure
```
.
├── api/
│ └── app.py # FastAPI app
├── src/
│ ├── train_classical.py # LR + SVM training
│ ├── inference_transformer.py
│ ├── evaluate.py
│ └── load_data.py
├── artifacts/ # Saved models + vectorizer
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## API Usage

### Endpoint
POST /predict


### Request

```json
{
  "text": "This paper proposes a neural architecture for..."
}
```

### Response

```json
{
  "prediction": "cs.NE",
  "confidence": 0.94
}
```

---

## Setup (Local)

git clone https://github.com/Priyanshi585-na/nlp_classifier.git
cd nlp_classifier

pip install -r requirements.txt
uvicorn api.app:app --reload

---

## Docker Setup
docker build -t nlp-classifier .
docker run -p 8000:8000 nlp-classifier

---

## Evaluation
Evaluation includes:

Accuracy
Precision / Recall / F1-score
Class-wise performance comparison

---

## Learnings
- TF-IDF struggles with semantic ambiguity
- Transformers capture contextual meaning
- Significant performance gains observed in overlapping classes
- Deployment introduces real-world considerations like latency and confidence handling

---

## Future Improvements
- Add top-k predictions
- Confidence calibration (temperature scaling)
- Batch inference optimization
- Model monitoring and logging

---

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
MIT License
