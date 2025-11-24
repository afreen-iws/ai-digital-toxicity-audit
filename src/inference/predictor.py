import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union

from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    ViTFeatureExtractor,
    ViTForImageClassification,
    BertModel,
    ViTModel,
)


class MultimodalFusionModel(nn.Module):
    """
    BERT (text) + ViT (image) late-fusion classifier.
    This matches the architecture used during training in Notebook 4.
    """

    def __init__(self, text_model_name: str, image_model_name: str, num_labels: int = 2):
        super().__init__()
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

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_embeds = text_outputs.pooler_output  # (batch, hidden)

        image_outputs = self.image_model(pixel_values=pixel_values)
        image_embeds = image_outputs.pooler_output  # (batch, hidden)

        combined = torch.cat((text_embeds, image_embeds), dim=1)
        logits = self.classifier(combined)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}


class ContentRiskPredictor:
    """
    Unified predictor that supports:
      - text-only prediction (BERT baseline)
      - image-only prediction (ViT baseline)
      - multimodal prediction (BERT + ViT fusion)
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

        # Text-only model
        self.text_model_dir = Path(text_model_dir)
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_dir)
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            self.text_model_dir
        ).to(self.device)
        self.text_model.eval()

        # Image-only model
        self.image_model_dir = Path(image_model_dir)
        self.image_feature_extractor = ViTFeatureExtractor.from_pretrained(
            self.image_model_dir
        )
        self.image_model = ViTForImageClassification.from_pretrained(
            self.image_model_dir
        ).to(self.device)
        self.image_model.eval()

        # Multimodal fusion model
        self.mm_dir = Path(multimodal_dir)
        self.mm_tokenizer = AutoTokenizer.from_pretrained(self.mm_dir)
        self.mm_feature_extractor = ViTFeatureExtractor.from_pretrained(self.mm_dir)

        self.mm_model = MultimodalFusionModel(
            text_model_name=text_model_name,
            image_model_name=image_model_name,
            num_labels=2,
        )

        # Load trained weights (saved by Trainer)
        state_dict_path = self.mm_dir / "pytorch_model.bin"
        if state_dict_path.exists():
            state_dict = torch.load(state_dict_path, map_location=self.device)
            self.mm_model.load_state_dict(state_dict)
        self.mm_model.to(self.device)
        self.mm_model.eval()

        # Label mapping
        self.id2label = {0: "safe", 1: "hateful_or_risky"}

    @staticmethod
    def _softmax(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    # ---------- TEXT-ONLY ----------
    def predict_text(self, text: str) -> dict:
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

        pred_id = int(probs.argmax())
        return {
            "mode": "text",
            "input_text": text,
            "pred_label_id": pred_id,
            "pred_label": self.id2label[pred_id],
            "risk_score": float(probs[1]),  # probability of hateful_or_risky
            "probs": {
                "safe": float(probs[0]),
                "hateful_or_risky": float(probs[1]),
            },
        }

    # ---------- IMAGE-ONLY ----------
    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
        return img

    def predict_image(self, image: Union[str, Path, Image.Image]) -> dict:
        img = self._load_image(image)
        inputs = self.image_feature_extractor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

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
        )

        # Image
        img_inputs = self.mm_feature_extractor(images=img, return_tensors="pt")

        # Move to device
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}

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
