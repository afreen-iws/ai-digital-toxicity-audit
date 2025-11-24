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
      - multimodal prediction (logical fusion of text + image)

    This implementation:
      - Uses BERT for text, ViT for images.
      - Applies a rule-based boost for certain clearly hateful phrases.
      - Uses probability thresholds to decide if content is hateful or safe.
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
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            text_model_name,
            num_labels=2,
        )

        # Try loading fine-tuned weights if available
        text_ckpt = self.text_model_dir / "pytorch_model.bin"
        if text_ckpt.exists():
            try:
                state_dict = torch.load(text_ckpt, map_location="cpu")
                self.text_model.load_state_dict(state_dict, strict=False)
                print(f"✅ Loaded fine-tuned text weights from: {text_ckpt}")
            except Exception as e:
                print(f"⚠ Could not load fine-tuned text weights from {text_ckpt}: {e}")
        else:
            print(
                f"⚠ Text checkpoint not found at {text_ckpt}, "
                f"using base '{text_model_name}' weights."
            )

        self.text_model.to(self.device)
        self.text_model.eval()

        # ------------- IMAGE MODEL (ViT) -------------
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
            print(
                f"⚠ Image checkpoint not found at {image_ckpt}, "
                f"using base '{image_model_name}' weights."
            )

        self.image_model.to(self.device)
        self.image_model.eval()

        # ------------- THRESHOLDS -------------
        # Thresholds for deciding hateful vs safe.
        # 0.6 is moderately sensitive; rules will boost clearly hateful phrases.
        self.text_threshold = 0.6
        self.image_threshold = 0.6
        self.multi_threshold = 0.6

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

    # ---------- RULE-BASED RISK BOOST ----------
    def post_process_risk(self, text: str, probs_dict: dict) -> dict:
        """
        Boost the hateful_or_risky probability if strong hateful patterns
        appear in the text. This helps catch clear hate that the model
        underestimates.
        """
        text_lower = text.lower()

        hate_phrases = [
            "go back",
            "ruining our country",
            "ruining this country",
            "send them back",
            "they don't belong",
            "they dont belong",
            "not welcome here",
            "kick them out",
            "should leave",
            "should be removed",
            "invaders",
            "parasites",
            "terrorist",
            "terrorists",
            "blow up",
            "a day without a blast",
            "blast is a day wasted",
            "paedophile",
            "pedophile",
            "9 year old",
            "exterminate",
            "scum",
        ]

        for phrase in hate_phrases:
            if phrase in text_lower:
                # Boost to at least 0.9 if matched
                probs_dict["hateful_or_risky"] = max(
                    probs_dict["hateful_or_risky"], 0.9
                )

        # Re-normalize so safe + hateful still sum to 1 (optional but cleaner)
        total = probs_dict["safe"] + probs_dict["hateful_or_risky"]
        if total > 0:
            probs_dict["safe"] /= total
            probs_dict["hateful_or_risky"] /= total

        return probs_dict

    # ---------- TEXT-ONLY ----------
    def predict_text(self, text: str) -> dict:
        """
        Predict risk using the text-only classifier.
        Uses probability threshold + rule-based boosting.
        """
        encoded = self.text_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.text_model(**encoded)
            logits = outputs.logits
            probs = self._softmax(logits).cpu().numpy()[0]

        probs_dict = {
            "safe": float(probs[0]),
            "hateful_or_risky": float(probs[1]),
        }

        # Apply keyword-based boost
        probs_dict = self.post_process_risk(text, probs_dict)

        risk_score = probs_dict["hateful_or_risky"]
        pred_is_hate = risk_score >= self.text_threshold

        pred_id = 1 if pred_is_hate else 0
        pred_label = self.id2label[pred_id]

        return {
            "mode": "text",
            "input_text": text,
            "pred_label_id": pred_id,
            "pred_label": pred_label,
            "risk_score": float(risk_score),
            "probs": {
                "safe": float(probs_dict["safe"]),
                "hateful_or_risky": float(probs_dict["hateful_or_risky"]),
            },
        }

    # ---------- IMAGE-ONLY ----------
    def predict_image(self, image: Union[str, Path, Image.Image]) -> dict:
        """
        Predict risk using the image-only classifier.
        Uses probability threshold, no text rules here.
        """
        img = self._load_image(image)
        inputs = self.image_processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

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
        Multimodal prediction using a simple rule:

        1. Run text-only and image-only predictors.
        2. Take the maximum risk score across branches.
        3. If max risk >= multi_threshold → label as hateful_or_risky.

        Because predict_text already applies rule-based boosting,
        clearly hateful captions in memes are more likely to be caught.
        """
        text = (text or "").strip()

        text_result = None
        if len(text) > 0:
            text_result = self.predict_text(text)

        image_result = None
        if image is not None:
            image_result = self.predict_image(image)

        scores = []
        if text_result is not None:
            scores.append(text_result["risk_score"])
        if image_result is not None:
            scores.append(image_result["risk_score"])

        if len(scores) == 0:
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
            "risk_score": float(final_score),
            "text_result": text_result,
            "image_result": image_result,
        }
