import torch
import torch.nn as nn
from pathlib import Path
from typing import Union
from PIL import Image

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    ViTImageProcessor,
    ViTForImageClassification,
    BertModel,
    ViTModel,
)


class MultimodalFusionModel(nn.Module):
    """
    BERT (text) + ViT (image) late-fusion classifier.
    """

    def __init__(self, text_model_name: str, image_model_name: str, num_labels: int = 2):
        super().__init__()

        self.text_model = BertModel.from_pretrained(text_model_name, local_files_only=True)
        self.image_model = ViTModel.from_pretrained(image_model_name, local_files_only=True)

        text_hidden = self.text_model.config.hidden_size
        image_hidden = self.image_model.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(text_hidden + image_hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels),
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = text_outputs.pooler_output

        image_outputs = self.image_model(pixel_values=pixel_values)
        image_embeds = image_outputs.pooler_output

        combined = torch.cat((text_embeds, image_embeds), dim=1)

        return {"logits": self.classifier(combined)}


class ContentRiskPredictor:
    """
    Unified predictor for:
      - Text-only classification
      - Image-only classification
      - Multimodal fusion classification
    """

    def __init__(
        self,
        text_model_dir="models/checkpoints/bert_text_baseline",
        image_model_dir="models/checkpoints/vit_image_baseline",
        multimodal_dir="models/checkpoints/multimodal_vit_bert",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id2label = {0: "safe", 1: "hateful_or_risky"}

        # ---- TEXT MODEL ----
        self.text_model_dir = Path(text_model_dir)
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            self.text_model_dir, local_files_only=True
        )
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            self.text_model_dir, local_files_only=True
        ).to(self.device)
        self.text_model.eval()

        # ---- IMAGE MODEL ----
        self.image_model_dir = Path(image_model_dir)
        self.image_processor = ViTImageProcessor.from_pretrained(
            self.image_model_dir, local_files_only=True
        )
        self.image_model = ViTForImageClassification.from_pretrained(
            self.image_model_dir, local_files_only=True
        ).to(self.device)
        self.image_model.eval()

        # ---- MULTIMODAL MODEL ----
        self.mm_dir = Path(multimodal_dir)
        self.mm_tokenizer = AutoTokenizer.from_pretrained(self.mm_dir, local_files_only=True)
        self.mm_image_processor = ViTImageProcessor.from_pretrained(
            self.mm_dir, local_files_only=True
        )

        self.mm_model = MultimodalFusionModel(
            text_model_name=self.text_model_dir,
            image_model_name=self.image_model_dir,
            num_labels=2,
        )

        fusion_weights = self.mm_dir / "pytorch_model.bin"
        if fusion_weights.exists():
            state_dict = torch.load(fusion_weights, map_location=self.device)
            self.mm_model.load_state_dict(state_dict)
        else:
            print(f"⚠ Warning: No weights found in {fusion_weights}")

        self.mm_model.to(self.device)
        self.mm_model.eval()

    @staticmethod
    def _softmax(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    # ------- TEXT-ONLY -------
    def predict_text(self, text: str) -> dict:
        encoded = self.text_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.text_model(**encoded)
            probs = self._softmax(outputs.logits).cpu().numpy()[0]

        pred_id = int(probs.argmax())
        return {
            "mode": "text",
            "text": text,
            "pred_label": self.id2label[pred_id],
            "risk_score": float(probs[1]),
        }

    # ------- IMAGE-ONLY -------
    def predict_image(self, image: Union[str, Image.Image, Path]) -> dict:
        img = self._load_image(image)
        inputs = self.image_processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.image_model(**inputs)
            probs = self._softmax(outputs.logits).cpu().numpy()[0]

        pred_id = int(probs.argmax())
        return {
            "mode": "image",
            "pred_label": self.id2label[pred_id],
            "risk_score": float(probs[1]),
        }

    # ------- MULTIMODAL -------
    def predict_multimodal(self, text: str, image: Union[str, Image.Image, Path]) -> dict:
        img = self._load_image(image)

        text_inputs = self.mm_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

        img_inputs = self.mm_image_processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.mm_model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                pixel_values=img_inputs["pixel_values"],
            )
            probs = self._softmax(outputs["logits"]).cpu().numpy()[0]

        pred_id = int(probs.argmax())
        return {
            "mode": "multimodal",
            "text": text,
            "pred_label": self.id2label[pred_id],
            "risk_score": float(probs[1]),
        }

    # ---------- IMAGE LOADING ----------
    @staticmethod
    def _load_image(image_input):
        if isinstance(image_input, Image.Image):
            return image_input
        return Image.open(image_input).convert("RGB")
