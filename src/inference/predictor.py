import torch
from pathlib import Path
from typing import Union

from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    AutoModelForSequenceClassification,
    ViTForImageClassification,
)


class ContentRiskPredictor:
    """
    Unified predictor that supports:
      - text-only prediction
      - image-only prediction
      - multimodal prediction (simple logical fusion of text + image)

    NOTE (Option B):
    This version does NOT rely on a separate trained fusion head.
    Instead, it:
      - Uses the text model as a risk scorer for text.
      - Uses the image model as a risk scorer for images.
      - For multimodal, combines both via a simple rule:
            final_risk = max(text_risk, image_risk)
            if final_risk >= multi_threshold → hateful_or_risky
        This is robust and behaves more predictably in deployment.
    """

    def __init__(
        self,
        text_model_dir: Union[str, Path] = "models/checkpoints/bert_text_baseline",
        image_model_dir: Union[str, Path] = "models/checkpoints/vit_image_baseline",
        text_model_name: str = "bert-base-uncased",
        image_model_name: str = "google/vit-base-patch16-224-in21k",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id2label = {0: "safe", 1: "hateful_or_risky"}

        # Directories where fine-tuned checkpoints may exist
        self.text_model_dir = Path(text_model_dir)
        self.image_model_dir = Path(image_model_dir)

        # ------------- TEXT MODEL (BERT) -------------
        # Load tokenizer and base model from Hugging Face Hub
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            text_model_name,
            num_labels=2,
        )

        # If there is a fine-tuned checkpoint in text_model_dir, load it
        text_ckpt = self.text_model_dir / "pytorch_model.bin"
        if text_ckpt.exists():
            try:
                state_dict = torch.load(text_ckpt, map_location="cpu")
                self.text_model.load_state_dict(state_dict, strict=False)
                print(f"✅ Loaded fine-tuned text weights from: {text_ckpt}")
            except Exception as e:
                print(f"⚠ Could not load fine-tuned text weights from {text_ckpt}: {e}")
        else:
            print(f"⚠ Text checkpoint not found at {text_ckpt}, using base '{text_model_name}' weights.")

        self.text_model.to(self.device)
        self.text_model.eval()

        # ------------- IMAGE MODEL (ViT) -------------
        # Image processor and base model from Hub
        self.image_processor = AutoImageProcessor.from_pretrained(image_model_name)
        self.image_model = ViTForImageClassification.from_pretrained(
            image_model_name,
            num_labels=2,
        )

        image_ckpt = self.image_model_dir / "pytorch_model.bin"
        if image_ckpt.exists():
            try:
                state_dict = torch.load(image_ckpt, map_location="cpu")
                self.image_model.load_state_dict(state_dict, strict=False)
                print(f"✅ Loaded fine-tuned image weights from: {image_ckpt}")
            except Exception as e:
                print(f"⚠ Could not load fine-tuned image weights from {image_ckpt}: {e}")
        else:
            print(f"⚠ Image checkpoint not found at {image_ckpt}, using base '{image_model_name}' weights.")

        self.image_model.to(self.device)
        self.image_model.eval()

        # ------------- THRESHOLDS -------------
        # You can tweak these if needed after testing.
        self.text_threshold = 0.8   # probability threshold for text branch
        self.image_threshold = 0.6  # probability threshold for image branch
        self.multi_threshold = 0.8  # final threshold for multimodal fusion

    # ---------- UTILS ----------
    @staticmethod
    def _softmax(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def _load_image(image: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
        return img

    # ---------- TEXT-ONLY ----------
    def predict_text(self, text: str) -> dict:
        """
        Predict risk using the text-only classifier.
        Uses a probability threshold to decide hateful vs safe.
        """
        encoded = self.text_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.text_model(**encoded)
            logits = outputs.logits
            probs = self._softmax(logits).cpu().numpy()[0]

        risk_score = float(probs[1])  # probability of hateful_or_risky
        pred_is_hate = risk_score >= self.text_threshold

        pred_id = 1 if pred_is_hate else 0
        pred_label = self.id2label[pred_id]

        return {
            "mode": "text",
            "input_text": text,
            "pred_label_id": pred_id,
            "pred_label": pred_label,
            "risk_score": risk_score,
            "probs": {
                "safe": float(probs[0]),
                "hateful_or_risky": float(probs[1]),
            },
        }

    # ---------- IMAGE-ONLY ----------
    def predict_image(self, image: Union[str, Path, Image.Image]) -> dict:
        """
        Predict risk using the image-only classifier.
        Also uses a probability threshold to decide hateful vs safe.
        """
        img = self._load_image(image)
        inputs = self.image_processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.image_model(**inputs)
            logits = outputs.logits
            probs = self._softmax(logits).cpu().numpy()[0]

        risk_score = float(probs[1])
        pred_is_hate = risk_score >= self.image_threshold

        pred_id = 1 if pred_is_hate else 0
        pred_label = self.id2label[pred_id]

        return {
            "mode": "image",
            "pred_label_id": pred_id,
            "pred_label": pred_label,
            "risk_score": risk_score,
            "probs": {
                "safe": float(probs[0]),
                "hateful_or_risky": float(probs[1]),
            },
        }

    # ---------- MULTIMODAL (TEXT + IMAGE) ----------
    def predict_multimodal(
        self,
        text: str,
        image: Union[str, Path, Image.Image],
    ) -> dict:
        """
        Multimodal prediction using a simple, robust rule:

        1. Run text-only and image-only predictors.
        2. Take the maximum risk score across branches.
        3. If max risk >= multi_threshold → label as hateful_or_risky.

        This avoids relying on a possibly brittle fusion head and is
        easier to reason about for a portfolio demo.
        """
        text = (text or "").strip()

        # Run branches conditionally
        text_result = None
        if len(text) > 0:
            text_result = self.predict_text(text)

        image_result = None
        if image is not None:
            image_result = self.predict_image(image)

        # Collect available risk scores
        scores = []
        if text_result is not None:
            scores.append(text_result["risk_score"])
        if image_result is not None:
            scores.append(image_result["risk_score"])

        if len(scores) == 0:
            # No usable input; default safe
            final_score = 0.0
            final_id = 0
            final_label = self.id2label[final_id]
        else:
            final_score = max(scores)
            final_is_hate = final_score >= self.multi_threshold
            final_id = 1 if final_is_hate else 0
            final_label = self.id2label[final_id]

        return {
            "mode": "multimodal",
            "input_text": text,
            "pred_label_id": final_id,
            "pred_label": final_label,
            "risk_score": final_score,
            "text_result": text_result,
            "image_result": image_result,
        }
