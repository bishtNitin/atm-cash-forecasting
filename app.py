"""
Operations Dashboard for ATM Cash Forecasting
Streamlit application for Ops Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import yaml
import os
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_pipeline import ATMDataPipeline
from models.lightgbm_model import LightGBMForecaster, MultiATMLightGBM
from explainability import ATMExplainer, create_business_impact_message


# Page configuration
st.set_page_config(
    page_title="ATM Cash Forecasting Dashboard",
    page_icon="🏧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_config():
    """Load configuration file"""
    config_path = 'config/config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


@st.cache_data
def load_data():
    """Load and process ATM data"""
    # Check if data exists, otherwise generate sample data
    data_path = 'data/raw/atm_transactions.csv'
    
    if not os.path.exists(data_path):
        st.info("Generating sample data...")
        config = load_config()
        pipeline = ATMDataPipeline(config)
        pipeline.generate_sample_data('data/raw', n_atms=100, n_days=365)
    
    # Load data
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    return df


@st.cache_resource
def load_model_and_explainer(df):
    """Train model and create explainer"""
    config = load_config()
    
    # Initialize pipeline
    pipeline = ATMDataPipeline(config)
    
    # Process data
    df_clean = pipeline.clean_data(df, method='linear')
    df_features = pipeline.engineer_features(df_clean, target_col='cash_withdrawn')
    
    # Prepare training data
    feature_cols = [col for col in df_features.columns 
                   if col not in ['date', 'cash_withdrawn', 'atm_id', 'latitude', 'longitude']]
    
    # Remove NaN rows
    df_features = df_features.dropna(subset=feature_cols + ['cash_withdrawn'])
    
    # Train model
    lgb_config = config.get('models', {}).get('lightgbm', {})
    model_wrapper = MultiATMLightGBM(lgb_config, mode='global')
    model_wrapper.train(df_features, target_col='cash_withdrawn')
    
    # Create explainer
    X_sample = df_features[feature_cols].sample(n=min(100, len(df_features)), random_state=42)
    explainer = ATMExplainer(
        model_wrapper.global_model.model, 
        feature_cols,
        config.get('explainability', {})
    )
    explainer.calculate_shap_values(X_sample)
    
    return model_wrapper, explainer, df_features, feature_cols


def create_india_map(df, risk_column='stockout_risk'):
    """Create map of India with ATM risk indicators"""
    # Get latest date data
    latest_date = df['date'].max()
    df_latest = df[df['date'] == latest_date].copy()
    
    # Calculate risk (mock calculation for demo)
    np.random.seed(42)
    df_latest['stockout_risk'] = np.random.uniform(0, 1, len(df_latest))
    
    # Create map centered on India
    india_map = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        tiles='OpenStreetMap'
    )
    
    # Add ATM markers
    for idx, row in df_latest.iterrows():
        # Color based on risk
        if row['stockout_risk'] > 0.4:
            color = 'red'
            risk_level = 'HIGH'
        elif row['stockout_risk'] > 0.2:
            color = 'orange'
            risk_level = 'MEDIUM'
        else:
            color = 'green'
            risk_level = 'LOW'
        
        # Create popup
        popup_text = f"""
        <b>{row['atm_id']}</b><br>
        Risk Level: {risk_level}<br>
        Stockout Risk: {row['stockout_risk']:.1%}<br>
        Cash: ₹{row['cash_withdrawn']/100000:.2f}L
        """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            popup=folium.Popup(popup_text, max_width=200),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(india_map)
    
    return india_map, df_latest


def show_network_health(df):
    """View 1: Network Health Map"""
    st.markdown('<p class="big-font">🗺️ Network Health Overview</p>', unsafe_allow_html=True)
    
    # Create risk summary
    col1, col2, col3, col4 = st.columns(4)
    
    # Get latest data
    latest_date = df['date'].max()
    df_latest = df[df['date'] == latest_date]
    
    # Mock risk calculations for demo
    np.random.seed(42)
    risk_scores = np.random.uniform(0, 1, len(df_latest))
    
    high_risk_count = sum(risk_scores > 0.4)
    medium_risk_count = sum((risk_scores > 0.2) & (risk_scores <= 0.4))
    low_risk_count = sum(risk_scores <= 0.2)
    total_atms = len(df_latest)
    
    col1.metric("Total ATMs", total_atms)
    col2.metric("🔴 High Risk", high_risk_count)
    col3.metric("🟡 Medium Risk", medium_risk_count)
    col4.metric("🟢 Low Risk", low_risk_count)
    
    # Create and display map
    st.subheader("ATM Risk Distribution Map")
    india_map, df_with_risk = create_india_map(df)
    folium_static(india_map, width=1200, height=600)
    
    # Risk distribution table
    st.subheader("High Risk ATMs")
    high_risk_atms = df_with_risk[df_with_risk['stockout_risk'] > 0.4].sort_values(
        'stockout_risk', ascending=False
    )
    
    if len(high_risk_atms) > 0:
        display_cols = ['atm_id', 'cash_withdrawn', 'stockout_risk', 'latitude', 'longitude']
        high_risk_display = high_risk_atms[display_cols].copy()
        high_risk_display['cash_withdrawn'] = high_risk_display['cash_withdrawn'].apply(
            lambda x: f"₹{x/100000:.2f}L"
        )
        high_risk_display['stockout_risk'] = high_risk_display['stockout_risk'].apply(
            lambda x: f"{x:.1%}"
        )
        st.dataframe(high_risk_display.head(10), use_container_width=True)
    else:
        st.success("No high-risk ATMs found!")


def show_atm_detail(df, model_wrapper, explainer, df_features, feature_cols):
    """View 2: ATM Detail Page"""
    st.markdown('<p class="big-font">🏧 ATM Detail Analysis</p>', unsafe_allow_html=True)
    
    # ATM Selection
    atm_ids = sorted(df['atm_id'].unique())
    selected_atm = st.selectbox("Select ATM", atm_ids, index=0)
    
    # Date Selection
    max_date = df['date'].max()
    selected_date = st.date_input(
        "Select Date for Forecast",
        value=max_date,
        min_value=df['date'].min(),
        max_value=max_date + timedelta(days=30)
    )
    
    # Get ATM data
    atm_data = df_features[df_features['atm_id'] == selected_atm].copy()
    
    if len(atm_data) == 0:
        st.error(f"No data available for {selected_atm}")
        return
    
    # Get latest features for prediction
    latest_data = atm_data.iloc[-1:][feature_cols]
    
    # Make prediction
    prediction = model_wrapper.predict(latest_data)[0]
    prediction_lakhs = prediction / 100000
    
    # Calculate stockout risk (mock for demo)
    np.random.seed(hash(selected_atm) % (2**32))
    stockout_risk = np.random.uniform(0.1, 0.6)
    
    # Display headline
    st.markdown('<p class="medium-font">📊 Forecast Summary</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Recommended Load", f"₹{prediction_lakhs:.2f} Lakhs")
    col2.metric("Stockout Risk", f"{stockout_risk:.1%}", 
               delta="High" if stockout_risk > 0.4 else "Low")
    col3.metric("Date", selected_date.strftime('%Y-%m-%d'))
    
    # Business Impact Section
    st.markdown('<p class="medium-font">💼 Business Impact</p>', unsafe_allow_html=True)
    
    impact_message = create_business_impact_message(
        prediction, 
        stockout_risk,
        threshold=prediction * 0.8
    )
    st.markdown(impact_message)
    
    # SHAP Explanation Section
    st.markdown('<p class="medium-font">🎯 Drivers of Demand - Why This Forecast?</p>', unsafe_allow_html=True)
    
    # Calculate SHAP values for this prediction
    try:
        shap_values_single = explainer.explainer.shap_values(latest_data)[0]
        
        # Create waterfall chart
        fig = explainer.plot_waterfall(
            shap_values_single,
            latest_data.iloc[0],
            prediction,
            max_display=8
        )
        st.pyplot(fig)
        plt.close()
        
        # Natural language explanation
        st.markdown("### 📝 Explanation in Plain English")
        explanation = explainer.explain_prediction(
            selected_atm,
            str(selected_date),
            shap_values_single,
            latest_data.iloc[0],
            prediction
        )
        st.info(explanation)
        
    except Exception as e:
        st.warning(f"Could not generate SHAP explanation: {str(e)}")
    
    # Historical Trend
    st.markdown('<p class="medium-font">📈 Historical Trend</p>', unsafe_allow_html=True)
    
    # Plot historical cash withdrawn
    atm_history = df[df['atm_id'] == selected_atm].copy()
    atm_history = atm_history.sort_values('date')
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=atm_history['date'],
        y=atm_history['cash_withdrawn'] / 100000,
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue', width=2)
    ))
    
    # Add prediction point
    fig_trend.add_trace(go.Scatter(
        x=[selected_date],
        y=[prediction_lakhs],
        mode='markers',
        name='Forecast',
        marker=dict(color='red', size=12, symbol='star')
    ))
    
    fig_trend.update_layout(
        title=f'Cash Withdrawal Trend - {selected_atm}',
        xaxis_title='Date',
        yaxis_title='Cash Withdrawn (₹ Lakhs)',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Feature Importance for this ATM
    st.markdown('<p class="medium-font">🔍 Key Factors</p>', unsafe_allow_html=True)
    
    top_features = explainer.get_top_features(n=8)
    
    fig_importance = go.Figure(go.Bar(
        x=top_features['importance'],
        y=top_features['feature'],
        orientation='h',
        marker=dict(color='steelblue')
    ))
    
    fig_importance.update_layout(
        title='Top Features Driving Predictions',
        xaxis_title='Average Impact',
        yaxis_title='Feature',
        height=400
    )
    
    st.plotly_chart(fig_importance, use_container_width=True)


def main():
    """Main application"""
    st.title("🏧 ATM Cash Forecasting - Operations Dashboard")
    st.markdown("*AI-Powered Demand Forecasting for 10,000+ ATMs across India*")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    view = st.sidebar.radio(
        "Select View",
        ["Network Health", "ATM Detail Analysis"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard provides AI-powered cash demand forecasting "
        "for ATM operations teams. It uses machine learning to predict "
        "cash requirements and explain the key drivers behind each forecast."
    )
    
    # Load data
    try:
        with st.spinner("Loading data..."):
            df = load_data()
        
        with st.spinner("Training model and preparing explanations..."):
            model_wrapper, explainer, df_features, feature_cols = load_model_and_explainer(df)
        
        # Show selected view
        if view == "Network Health":
            show_network_health(df)
        else:
            show_atm_detail(df, model_wrapper, explainer, df_features, feature_cols)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
