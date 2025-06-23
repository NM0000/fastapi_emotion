from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from onnx_inference import EmotionClassifierONNX

# Load the classifier on startup
classifier = EmotionClassifierONNX(
    model_path="emotion_model.onnx",
    label_encoder_path="label_encoder.pkl"
)

# Define request body
class TextRequest(BaseModel):
    texts: List[str]

app = FastAPI(title="Emotion Classifier API")

@app.get("/")
def home():
    return {"message": "Emotion classifier is up and running!"}

@app.post("/predict")
def predict_emotions(request: TextRequest):
    labels = classifier.predict(request.texts)
    return {"predictions": labels.tolist() if hasattr(labels, 'tolist') else list(labels)}
