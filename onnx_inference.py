import pickle
import numpy as np
from transformers import DistilBertTokenizer
import onnxruntime as ort

class EmotionClassifierONNX:
    def __init__(self, model_path, label_encoder_path, max_len=32):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_len = max_len
        with open(label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

    def predict(self, texts):
        encodings = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='np'
        )
        inputs = {
            "input_ids": encodings['input_ids'],
            "attention_mask": encodings['attention_mask'],
        }
        logits = self.session.run(None, inputs)[0]
        preds = np.argmax(logits, axis=1)
        return self.label_encoder.inverse_transform(preds)
