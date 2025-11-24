import torch
import torch.nn as nn
from pathlib import Path
from typing import Union

from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    AutoModelForSequenceClassification,
    ViTForImageClassification,
    BertModel,
    ViTModel,
)


class MultimodalFusionModel(nn.Module):
    """
    BERT (text) + ViT (image) late-fusion classifier.
    Uses Hugging Face base models, then can load fine-tuned weights separately.
    """

    def __init__(self, text_model_name: str, image_model_name: str, num_labels: int = 2):
        super().__init__()

        # Load base pretrained backbones from Hugging Face Hub
        self.text_model = BertModel.from_pretrained(text_model_name)
        self.image_model = ViTModel.from_pretrained(image_model_name)

        text_hidden = self.text_model.config.hidden_size
        image_hidden = self.image_model.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(text_hidden + image_hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels),
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        # Text branch
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = text_outputs.pooler_output  # (batch, hidden)

        # Image branch
        image_outputs = self.image_model(pixel_values=pixel_values)
        image_embeds = image_outputs.pooler_output  # (batch, hidden)

        # Late fusion
        combined = torch.cat((text_embeds, image_embeds), dim=1)
        logits = self.classifier(combined)
        return {"logits": logits}


class ContentRiskPredictor:
    """
    Unified predictor that supports:
      - Text-only prediction (BERT baseline)
      - Image-only prediction (ViT baseline)
      - Multimodal prediction (BERT + ViT fusion)
    """

    def __init__(
        self,
        text_model_dir: Union[str, Path] = "models/checkpoints/bert_text_baseline",
        image_model_dir: Union[str, Path] = "models/checkpoints/vit_image_baseline",
        multimodal_dir: Union[str, Path] = "models/checkpoints/multimodal_vit_bert",
        text_model_name: str = "bert-base-uncased",
        image_model_name: str = "google/vit-base-patch16-224-in21k",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id2label = {0: "safe", 1: "hateful_or_risky"}

        # Paths to local fine-tuned checkpoints (optional)
        self.text_model_dir = Path(text_model_dir)
        self.image_model_dir = Path(image_model_dir)
        self.mm_dir = Path(multimodal_dir)

        # -------- TEXT-ONLY MODEL (BERT) --------
        # Always load tokenizer from Hugging Face Hub
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)

        # Base model from Hub
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            text_model_name,
            num_labels=2,
        )

        # If fine-tuned weights exist locally, load them
        text_ckpt = self.text_model_dir / "pytorch_model.bin"
        if text_ckpt.exists():
            state_dict = torch.load(text_ckpt, map_location="cpu")
            self.text_model.load_state_dict(state_dict, strict=False)
        else:
            print(f"⚠ Text checkpoint not found at {text_ckpt}, using base '{text_model_name}' weights.")

        self.text_model.to(self.device)
        self.text_model.eval()

        # -------- IMAGE-ONLY MODEL (ViT) --------
        # Image processor from Hub
        self.image_processor = AutoImageProcessor.from_pretrained(image_model_name)

        # Base ViT classifier from Hub (will adapt head to 2 labels)
        self.image_model = ViTForImageClassification.from_pretrained(
            image_model_name,
            num_labels=2,
        )

        image_ckpt = self.image_model_dir / "pytorch_model.bin"
        if image_ckpt.exists():
            state_dict = torch.load(image_ckpt, map_location="cpu")
            self.image_model.load_state_dict(state_dict, strict=False)
        else:
            print(f"⚠ Image checkpoint not found at {image_ckpt}, using base '{image_model_name}' weights.")

        self.image_model.to(self.device)
        self.image_model.eval()

        # -------- MULTIMODAL FUSION MODEL --------
        # Re-use the same tokenizer and processor for multimodal
        self.mm_tokenizer = self.text_tokenizer
        self.mm_processor = self.image_processor

        # Base fusion model with backbone models from Hugging Face Hub
        self.mm_model = MultimodalFusionModel(
            text_model_name=text_model_name,
            image_model_name=image_model_name,
            num_labels=2,
        )

        mm_ckpt = self.mm_dir / "pytorch_model.bin"
        if mm_ckpt.exists():
            state_dict = torch.load(mm_ckpt, map_location="cpu")
            self.mm_model.load_state_dict(state_dict, strict=False)
        else:
            print(f"⚠ Multimodal checkpoint not found at {mm_ckpt}, using unfine-tuned fusion model.")

        self.mm_model.to(self.device)
        self.mm_model.eval()

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

        pred_id = int(probs.argmax())
        return {
            "mode": "text",
            "input_text": text,
            "pred_label_id": pred_id,
            "pred_label": self.id2label[pred_id],
            "risk_score": float(probs[1]),
            "probs": {
                "safe": float(probs[0]),
                "hateful_or_risky": float(probs[1]),
            },
        }

    # ---------- IMAGE-ONLY ----------
    def predict_image(self, image: Union[str, Path, Image.Image]) -> dict:
        img = self._load_image(image)
        inputs = self.image_processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.image_model(**inputs)
            logits = outputs.logits
            probs = self._softmax(logits).cpu().numpy()[0]

        pred_id = int(probs.argmax())
        return {
            "mode": "image",
            "pred_label_id": pred_id,
            "pred_label": self.id2label[pred_id],
            "risk_score": float(probs[1]),
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
        img = self._load_image(image)

        # Text
        text_inputs = self.mm_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        # Image
        img_inputs = self.mm_processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.mm_model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                pixel_values=img_inputs["pixel_values"],
            )
            logits = outputs["logits"]
            probs = self._softmax(logits).cpu().numpy()[0]

        pred_id = int(probs.argmax())
        return {
            "mode": "multimodal",
            "input_text": text,
            "pred_label_id": pred_id,
            "pred_label": self.id2label[pred_id],
            "risk_score": float(probs[1]),
            "probs": {
                "safe": float(probs[0]),
                "hateful_or_risky": float(probs[1]),
            },
        }
