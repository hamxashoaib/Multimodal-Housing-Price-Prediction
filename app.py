"""
Multimodal Housing Price Predictor - Streamlit App
DevelopersHub Corporation AI/ML Internship

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ── Load pipeline ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    try:
        bundle = joblib.load('multimodal_pipeline.pkl')
        return bundle, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_resnet():
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet.eval()
    return resnet

IMG_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏠 Multimodal Housing Price Predictor")
st.markdown(
    "Predicts house price using **both a house photo (CNN features)** and "
    "**structured data** — combining two modalities for better accuracy."
)
st.markdown("---")

bundle, error = load_pipeline()

if error:
    st.error(f"⚠️ Could not load pipeline: `{error}`")
    st.info("Run the notebook first to generate `multimodal_pipeline.pkl`.")
    st.stop()

resnet = load_resnet()
st.success("✅ Multimodal pipeline loaded!")
st.markdown("---")

# ── Input ─────────────────────────────────────────────────────────────────────
st.subheader("📋 House Details")

col1, col2, col3 = st.columns(3)

with col1:
    area_sqft      = st.number_input("Area (sqft)", 500, 6000, 1800, step=50)
    bedrooms       = st.slider("Bedrooms", 1, 6, 3)
    bathrooms      = st.select_slider("Bathrooms", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], value=2.0)
    garage_cars    = st.slider("Garage Cars", 0, 4, 1)

with col2:
    year_built     = st.slider("Year Built", 1950, 2023, 2005)
    lot_area       = st.number_input("Lot Area (sqft)", 2000, 25000, 8000, step=500)
    overall_qual   = st.slider("Overall Quality (1–10)", 1, 10, 7)
    location_score = st.slider("Location Score (1–10)", 1.0, 10.0, 6.0, step=0.5)

with col3:
    has_pool       = st.selectbox("Has Pool?", ["No", "Yes"])
    has_basement   = st.selectbox("Has Basement?", ["Yes", "No"])
    st.markdown("**Upload House Image**")
    uploaded_img   = st.file_uploader("Upload a house photo (.jpg / .png)",
                                       type=["jpg", "jpeg", "png"])
    if uploaded_img:
        st.image(uploaded_img, caption="Uploaded House", use_column_width=True)

st.markdown("---")
predict_btn = st.button("🔍 Predict Price", type="primary")

# ── Predict ───────────────────────────────────────────────────────────────────
if predict_btn:
    # Build tabular row
    tabular_input = np.array([[
        area_sqft, bedrooms, bathrooms, garage_cars,
        year_built, lot_area, overall_qual, location_score,
        1.0 if has_pool == "Yes" else 0.0,
        1.0 if has_basement == "Yes" else 0.0
    ]])

    # Extract image features
    if uploaded_img:
        img = Image.open(uploaded_img).convert('RGB')
    else:
        # Fallback: blank image
        img = Image.new('RGB', (224, 224), color=(180, 160, 120))
        st.info("No image uploaded — using a placeholder. Upload a real photo for best results.")

    with torch.no_grad():
        img_tensor = IMG_TRANSFORMS(img).unsqueeze(0)
        img_feat   = resnet(img_tensor).numpy()   # (1, 512)

    # PCA + scale + predict
    pca    = bundle['pca']
    scaler = bundle['scaler']
    model  = bundle['model']

    img_feat_reduced = pca.transform(img_feat)                     # (1, 64)
    fused            = np.hstack([tabular_input, img_feat_reduced]) # (1, 74)
    fused_scaled     = scaler.transform(fused)
    predicted_price  = model.predict(fused_scaled)[0]

    # Result
    st.subheader("Prediction Result")
    st.markdown(
        f"""
        <div style="background:#eafaf1; border-left:5px solid #27AE60;
                    padding:22px 26px; border-radius:8px; margin-bottom:16px;">
            <h2 style="color:#27AE60; margin:0;">🏷️ Estimated Price</h2>
            <h1 style="color:#1a5276; margin:8px 0 0 0;">${predicted_price:,.0f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Feature summary
    with st.expander("See input summary"):
        st.dataframe(pd.DataFrame({
            'Feature': ['Area (sqft)', 'Bedrooms', 'Bathrooms', 'Garage Cars',
                        'Year Built', 'Lot Area', 'Overall Quality',
                        'Location Score', 'Has Pool', 'Has Basement'],
            'Value': [area_sqft, bedrooms, bathrooms, garage_cars,
                      year_built, lot_area, overall_qual, location_score,
                      has_pool, has_basement]
        }))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Task 3** — AI/ML Internship  
    DevelopersHub Corporation

    **How it works:**
    1. House photo → **ResNet-18** → 512-dim features → **PCA** → 64 dims
    2. Tabular data (10 features) → merged with image features
    3. Combined → **Gradient Boosting Regressor** → price

    **Metrics:**
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)

    **Multimodal beats tabular-only** by combining visual + structured signals.

    ---
    Built by Hamza Shoaib  
    """)
