%%writefile app/streamlit_app.py
import streamlit as st
from PIL import Image
from pathlib import Path
import torch

from src.inference.predictor import ContentRiskPredictor

# Load predictor once
@st.cache_resource
def load_predictor():
    return ContentRiskPredictor()

predictor = load_predictor()

st.title("🛡️ Content Risk Detector for Social Media & Advertising")
st.write(
    "This ML-based AI system analyzes **text**, **image**, or **both** to detect "
    "potentially hateful or risky content. It supports multimodal classification "
    "using BERT (text) and ViT (image) fusion."
)

mode = st.radio(
    "Select Input Mode:",
    options=["Text-only", "Image-only", "Multimodal (Image + Text)"]
)

# ---------- TEXT-ONLY MODE ----------
if mode == "Text-only":
    text_input = st.text_area("Enter content text here:", height=150)
    if st.button("Analyze Text"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            result = predictor.predict_text(text_input)
            st.subheader(f"Prediction: {result['pred_label']}")
            st.write(f"**Risk Score (prob of hateful/risky):** {result['risk_score']:.4f}")
            st.json(result)

# ---------- IMAGE-ONLY MODE ----------
elif mode == "Image-only":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        if st.button("Analyze Image"):
            result = predictor.predict_image(img)
            st.subheader(f"Prediction: {result['pred_label']}")
            st.write(f"**Risk Score (prob of hateful/risky):** {result['risk_score']:.4f}")
            st.json(result)

# ---------- MULTIMODAL MODE (IMAGE + TEXT) ----------
elif mode == "Multimodal (Image + Text)":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    text_input = st.text_area("Enter associated text/caption (optional):", height=120)

    if uploaded_image is not None:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Image + Text"):
            if not text_input.strip():
                st.warning(
                    "A multimodal prediction typically requires both image and text. "
                    "You can still proceed with only the image, but consider adding text for better accuracy."
                )

            result = predictor.predict_multimodal(text_input or "", img)
            st.subheader(f"Prediction: {result['pred_label']}")
            st.write(f"**Risk Score (prob of hateful/risky):** {result['risk_score']:.4f}")
            st.json(result)
