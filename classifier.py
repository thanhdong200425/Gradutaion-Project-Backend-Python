import os
from typing import TypedDict

import py_vncorenlp
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class PredictionScores(TypedDict):
    easy: float
    medium: float
    hard: float


class PredictionResult(TypedDict):
    difficulty: str
    confidence: float
    scores: PredictionScores


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VNCORENLP_DIR = os.path.join(_BASE_DIR, "vncorenlp")
MODEL_PATH = os.path.join(VNCORENLP_DIR, "phobert-difficulty-final")
LABELS = {0: "easy", 1: "medium", 2: "hard"}


class DifficultyClassifier:
    def __init__(self) -> None:
        print("Loading word segmenter...")
        self.segmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"], save_dir=VNCORENLP_DIR
        )

        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        self.model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        print(f"Model ready on {self.device}")

    def predict(self, questions: list[str]) -> list[PredictionResult]:
        result: list[PredictionResult] = []
        for question in questions:
            # 1: Word segmentation
            segmented = " ".join(self.segmenter.word_segment(question))

            # 2: Tokenizer
            inputs = self.tokenizer(
                segmented,
                max_length=256,
                truncation=True,  # ensure the input just has max length is 256
                padding="max_length",  # add padding if the segmented is not have 256
                return_tensors="pt",  # return pytorch tensors
            )

            # 3: Inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 4: Get the proabilities
            probs = F.softmax(outputs.logits, dim=-1)[0]
            pred_id = torch.argmax(probs).item()
            result.append(
                {
                    "difficulty": LABELS[pred_id],
                    "confidence": round(probs[pred_id].item(), 4),
                    "scores": {
                        "easy": round(probs[0].item(), 4),
                        "medium": round(probs[1].item(), 4),
                        "hard": round(probs[2].item(), 4),
                    },
                }
            )

        return result


classifier = DifficultyClassifier()
