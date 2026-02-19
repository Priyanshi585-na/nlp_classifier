import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


tokenizer = AutoTokenizer.from_pretrained('priyanshisalujaaa112/nlp-distilbert-arxiv')
model = AutoModelForSequenceClassification.from_pretrained('priyanshisalujaaa112/nlp-distilbert-arxiv')

model.eval()

label_map = {
    0: "cs.CV",
    1: "cs.AI",
    2: "cs.SY",
    3: "cs.DS",
    4: "cs.NE"
}

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors = "pt",
        truncation = True,
        padding = True,
        max_length = 256
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim = 1)
        pred= torch.argmax(probs, dim = 1).item()

    return{
        "prediction":label_map[pred],
        "confidence":float(probs[0][pred])
    }
