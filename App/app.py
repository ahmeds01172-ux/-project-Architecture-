import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Egypt Real Estate Appraiser",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    .main-header { font-size: 2.8em; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 5px; }
    .price-output {
        font-size: 2.8em; font-weight: bold; color: #27ae60; text-align: center;
        padding: 25px; border: 4px solid #27ae60; border-radius: 12px;
        background-color: #f0f9f0; margin: 20px 0;
    }
    .metric-box {
        background-color: #f8f9fa; padding: 18px; border-radius: 10px;
        border-left: 6px solid #1f77b4; text-align: center; height: 100%;
    }
    .metric-label { font-size: 0.95em; color: #555; margin-bottom: 5px; }
    .metric-value { font-size: 1.4em; font-weight: bold; color: #1f77b4; }
    .viz-header { font-size: 1.4em; font-weight: bold; color: #1f77b4; margin: 20px 0 10px 0; }
</style>
""", unsafe_allow_html=True)

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model():
    model_path = 'models/best_pipeline.pkl'
    if not os.path.exists(model_path):
        st.error("❌ Model not found! Please train the model first.")
        st.stop()
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

pipeline = load_model()

# ====================== LOAD DATA FOR VISUALIZATIONS ======================
@st.cache_data
def load_data():
    return pd.read_pickle('data/cleaned_data.pkl')

df = load_data()

# ====================== HEADER ======================
st.markdown('<p class="main-header">🏠 Egypt Real Estate Appraiser</p>', unsafe_allow_html=True)
st.markdown("**AI-Powered Instant Property Valuation for Egypt**", unsafe_allow_html=True)
st.divider()

# ====================== INPUT SECTION ======================
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📋 Property Details")
    location_group = st.selectbox("📍 Location Area", 
        ["East Cairo", "West Cairo", "6 October City", "North Coast", "Red Sea", "Other"])
    property_type = st.selectbox("🏢 Property Type", 
        ["Apartment", "Villa", "Chalet", "Duplex", "Penthouse", "Studio", "Townhouse", "Twin House"])
    size_sqm = st.number_input("📐 Size (sqm)", min_value=30, max_value=2000, value=140, step=5)
    bedrooms = st.number_input("🛏️ Bedrooms", min_value=0, max_value=10, value=2, step=1)

with col2:
    st.subheader("🔧 Additional Details")
    bathrooms = st.number_input("🚿 Bathrooms", min_value=0, max_value=8, value=1, step=1)
    payment_method = st.selectbox("💳 Payment Method", ["Cash", "Installments"])
    luxury = st.checkbox("✨ Luxury Property (Premium location/amenities)", value=False)

# ====================== PREDICTION ======================
st.divider()
if st.button("🎯 Estimate Price", type="primary", use_container_width=True):
    
    input_df = pd.DataFrame({
        'location_group': [location_group],
        'type': [property_type],
        'payment_method': [payment_method],
        'size_sqm': [float(size_sqm)],
        'bedrooms_num': [float(bedrooms)],
        'bathrooms': [float(bathrooms)]
    })
    
    prediction = pipeline.predict(input_df)[0]
    
    st.success("✅ Prediction Generated!")
    st.markdown(f'<p class="price-output">{prediction:,.0f} EGP</p>', unsafe_allow_html=True)
    
    # Metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Price per sqm</div>
            <div class="metric-value">{(prediction / size_sqm):,.0f} EGP</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Total Size</div>
            <div class="metric-value">{size_sqm:,} sqm</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Property Type</div>
            <div class="metric-value">{property_type}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.balloons()

# ====================== VISUALIZATIONS SECTION ======================
st.divider()
st.markdown('<p class="viz-header">📊 Exploratory Data Analysis Visualizations</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Price Distribution", "By Property Type", "Correlation", "Top Locations"])

with tab1:
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.image('reports/price_distribution.png', use_column_width=True)
    with col_v2:
        st.image('reports/price_log_distribution.png', use_column_width=True)

with tab2:
    st.image('reports/price_by_type.png', use_column_width=True)

with tab3:
    st.image('reports/correlation_heatmap.png', use_column_width=True)

with tab4:
    st.image('reports/top_locations.png', use_column_width=True)

# Extra visualizations in expander
with st.expander("📈 More Visualizations"):
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.image('reports/size_vs_price.png', use_column_width=True)
    with col_e2:
        st.image('reports/type_distribution.png', use_column_width=True)

# ====================== FOOTER ======================
st.divider()
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.info("📊 Dataset: 16,000+ Egyptian Real Estate Listings")
with col_f2:
    st.info("🤖 Model: Ensemble Machine Learning Pipeline")
with col_f3:
    st.info("📈 Accuracy: R² ≈ 0.52 on test data")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Egypt Real Estate Appraiser v1.0")