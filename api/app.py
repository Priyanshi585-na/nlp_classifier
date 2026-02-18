from fastapi import FastAPI
from pydantic import BaseModel
from src.inference_transformer import predict

app = FastAPI()

class AbstractInput(BaseModel):
    text : str

@app.post("/predict")
def classify_abstract(input: AbstractInput):
    return predict(input.text)
    