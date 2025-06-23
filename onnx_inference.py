import pickle
import numpy as np
from transformers import DistilBertTokenizer
import onnxruntime as ort
from huggingface_hub import hf_hub_download

class EmotionClassifierONNX:
    def __init__(self, model_path=None, label_encoder_path=None, max_len=32):
        # ✅ Download the ONNX model locally from Hugging Face Hub
        self.model_path = hf_hub_download(
            repo_id="nishil00/model_emotion",
            filename="emotion_model.onnx",
            local_dir=".",
            local_dir_use_symlinks=False
        )
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])

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
