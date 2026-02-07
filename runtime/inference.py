import json
import os
import joblib
import numpy as np


class StaticModel:
    def __init__(self, model_path=None, labelmap_path=None):
        # Resolve paths relative to the project root (…/SignToSpeech)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        if model_path is None:
            model_path = os.path.join(project_root, "models", "static_lr.joblib")
        elif not os.path.isabs(model_path):
            model_path = os.path.join(project_root, model_path)

        if labelmap_path is None:
            labelmap_path = os.path.join(project_root, "models", "label_map.json")
        elif not os.path.isabs(labelmap_path):
            labelmap_path = os.path.join(project_root, labelmap_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not os.path.exists(labelmap_path):
            raise FileNotFoundError(f"Label map not found: {labelmap_path}")

        self.model = joblib.load(model_path)

        with open(labelmap_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # JSON keys are strings → convert to int
        self.labelmap = {int(k): v for k, v in raw.items()}

    def predict(self, vec: np.ndarray):
        X = vec.reshape(1, -1)
        probs = self.model.predict_proba(X)[0]
        idx = int(np.argmax(probs))
        return self.labelmap[idx], float(probs[idx])
