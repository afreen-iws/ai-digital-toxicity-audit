import streamlit as st
from PIL import Image

from src.inference.predictor import ContentRiskPredictor


# Load predictor once and cache it across reruns
@st.cache_resource
def load_predictor():
    return ContentRiskPredictor()


predictor = load_predictor()

st.set_page_config(
    page_title="Content Risk Detector",
    page_icon="🛡️",
    layout="centered",
)

st.title("🛡️ Content Risk Detector for Social Media & Advertising")

st.markdown(
    """
This application uses a **multimodal ML-based AI system** to assess whether content is
potentially **safe** or **hateful/risky**.

It supports three modes:
- **Text-only**: analyze captions, comments, or posts  
- **Image-only**: analyze images or memes  
- **Multimodal (Image + Text)**: combine image and text context for better detection  
"""
)

mode = st.radio(
    "Select input mode:",
    options=["Text-only", "Image-only", "Multimodal (Image + Text)"],
    index=0,
)


def display_result(result: dict):
    """
    Helper function to display prediction result from ContentRiskPredictor.
    """
    label = result["pred_label"]
    risk_score = result["risk_score"]

    if label == "hateful_or_risky":
        st.error(f"Prediction: **{label}**")
    else:
        st.success(f"Prediction: **{label}**")

    st.write(f"**Risk Score (probability of hateful/risky):** `{risk_score:.4f}`")

    with st.expander("Show raw prediction details"):
        st.json(result)


# ------------------- TEXT-ONLY MODE -------------------
if mode == "Text-only":
    st.subheader("Text-only Analysis")

    text_input = st.text_area(
        "Enter content text",
        placeholder="Type or paste a caption, comment, or post here...",
        height=150,
    )

    if st.button("Analyze Text"):
        if not text_input.strip():
            st.warning("Please enter some text before running the analysis.")
        else:
            result = predictor.predict_text(text_input)
            display_result(result)

# ------------------- IMAGE-ONLY MODE -------------------
elif mode == "Image-only":
    st.subheader("Image-only Analysis")

    uploaded_image = st.file_uploader(
        "Upload an image (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_image is not None:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Image"):
            result = predictor.predict_image(img)
            display_result(result)
    else:
        st.info("Please upload an image to analyze.")

# -------------- MULTIMODAL (IMAGE + TEXT) MODE --------------
elif mode == "Multimodal (Image + Text)":
    st.subheader("Multimodal Analysis (Image + Text)")

    uploaded_image = st.file_uploader(
        "Upload an image (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        key="multimodal_image",
    )

    text_input = st.text_area(
        "Optional: enter associated text/caption",
        placeholder="Add accompanying text to provide more context (recommended).",
        height=120,
    )

    if uploaded_image is not None:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Image + Text"):
            # Text can be empty string; the predictor will still run
            result = predictor.predict_multimodal(text_input or "", img)
            display_result(result)
    else:
        st.info("Please upload an image to run multimodal analysis.")
