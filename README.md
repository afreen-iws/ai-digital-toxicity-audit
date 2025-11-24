**Content Risk Detector for Social Media & Advertising**

Multimodal AI System for Detecting Toxic / Hateful Content in Images, Text, and Memes

**Live Demo** (Streamlit App): https://ai-digital-toxicity-audit.streamlit.app/

**Supported Inputs**: Text 📝 | Image 🖼️ | Image + Text (Memes) 🎯

**Model Type**: ML-based Multimodal Fusion (Vision Transformer + BERT)

**Project Overview**

This project builds a machine learning–based AI system that detects whether social media content (text, image, or meme with both) is:

✔️ Safe
🚩 Hateful / Toxic / Risky

It is designed as a portfolio-ready end-to-end ML project — including:
| Component     | Status                                                         |
| ------------- | -------------------------------------------------------------- |
| Dataset       | Facebook Hateful Memes (public)                                |
| Model Types   | Text-only (BERT), Image-only (ViT), Multimodal Fusion          |
| Training      | Done in Google Colab using PyTorch + Hugging Face Transformers |
| Evaluation    | Accuracy, F1, Confusion Matrix, Error Analysis                 |
| Deployment    | Streamlit Cloud (Live Web App)                                 |
| ML Techniques | Late Fusion Architecture, Attention-based Learning             |
| Front-End     | Streamlit UI — supports text-only, image-only, and multimodal  |

**Why This Project Is Unique**

Most hate speech detectors only analyze text — this project detects toxicity even when it's hidden in images, memes, or sarcastic visuals.

This model detects harmful intent even when:

🖼️ The text alone seems harmless, but the image adds context.
📝 The image alone seems harmless, but text adds hateful meaning.
🎯 Combined image + text reveals true toxicity — only a multimodal model can detect this.

**Business Use Case**:
This model can be used for content moderation, brand safety auditing, advertising screening, or policy compliance on Facebook, Instagram, TikTok, and YouTube.



**Dataset**
| Feature | Description                                    |
| ------- | ---------------------------------------------- |
| Name    | Facebook Hateful Memes                         |
| Type    | Multimodal (Images + Text)                     |
| Size    | ~10,000 memes                                  |
| Labels  | 0 = Safe, 1 = Hateful                          |
| Format  | JSONL files with image filename + text + label |

**Sample JSONL Record**
{
  "id": 10244,
  "img": "10244.png",
  "text": "Go back to where you came from",
  "label": 1
}

⚙️ **Setup Instructions (Google Colab)**
**1. Clone the repo**
!git clone https://github.com/afreen-iws/ai-digital-toxicity-audit.git
%cd ai-digital-toxicity-audit

**2.Install dependencies**
!pip install -r requirements.txt

**3. Mount Google Drive (to access model checkpoints)**
from google.colab import drive
drive.mount('/content/drive')

**Model Architectures**

1. **Text-Only Model (BERT)**
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

2. **Image-only Model (ViT)**
from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2
)

3. **Multimodal Fusion Model (Late Fusion)**
class MultimodalFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = BertModel.from_pretrained("bert-base-uncased")
        self.image_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.classifier = nn.Sequential(
            nn.Linear(768 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

**Evaluation Summary**
Model	Accuracy	F1-Score
BERT (Text-only)	0.56	0.32
ViT (Image-only)	0.58	0.38
🏆 Fusion Model	0.59	0.46

**The multimodal model performs best, especially in detecting context-based hate.**

🧪 **Sample Output**
Input Type	Prediction
"They don't belong here"	🔴 Hateful
Meme of cartoon mocking ethnicity	🔴 Hateful
Positive quote image	🟢 Safe
Image + text: "Go back to your country"	🚨 High Risk

**Streamlit Deployment**
**How inference works in streamlit_app.py**

from src.inference.predictor import ContentRiskPredictor

predictor = ContentRiskPredictor()

result = predictor.predict_multimodal(
    text="Go back to your country",
    image=uploaded_image
)

**Business Applications**

✔ Brand safety for advertising
✔ Content moderation for social platforms
✔ Marketing compliance audits
✔ Online abuse monitoring
✔ Hate speech detection in memes

**Key Learnings**

🔹 How to combine BERT and ViT using PyTorch
🔹 Late fusion design for multimodal ML
🔹 Training, evaluation, and logging in Transformer's Trainer API
🔹 Deploy deep learning models using Streamlit Cloud
🔹 Handle inference for text, image, and combined input


