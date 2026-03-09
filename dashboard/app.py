"""
Driver Drowsiness Detection - Interactive Dashboard
====================================================
Visual interface for real-time driver safety monitoring
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import time

# Configuration
API_URL = "http://localhost:8000"
API_KEY = "dev_secure_key_123"
HEADERS = {"x-api-key": API_KEY}

# Page config
st.set_page_config(
    page_title="Driver Safety Monitor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Enhanced Modern Theme
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 30px;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric label {
        color: white !important;
        font-weight: bold;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 2rem !important;
    }
    div[data-testid="stExpander"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        border: none;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 15px 30px;
        border-radius: 10px;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🚗 Driver Drowsiness Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">🧠 AI-Powered Real-Time Safety Monitoring | ✨ Enhanced Confidence Predictions</p>', unsafe_allow_html=True)

# Create tabs for different features
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyze Driver", "📊 System Metrics", "🔧 Diagnostics", "📈 Performance"])

# TAB 1: Driver Analysis
with tab1:
st.sidebar.header("📊 Driver Status Input")

speed = st.sidebar.slider("Speed (km/h)", 0, 200, 60)
alertness = st.sidebar.slider("Alertness Level", 0.0, 1.0, 0.8, 0.01)
seatbelt = st.sidebar.selectbox("Seatbelt", [1, 0], format_func=lambda x: "On" if x == 1 else "Off")
hr = st.sidebar.slider("Heart Rate (bpm)", 30, 200, 75)
fatigue = st.sidebar.slider("Fatigue Level", 0, 10, 3)
speed_change = st.sidebar.slider("Speed Variability", 0.0, 20.0, 5.0)
prev_alertness = st.sidebar.slider("Previous Alertness", 0.0, 1.0, 0.85, 0.01)

analyze_button = st.sidebar.button("🔍 Analyze Driver", type="primary", use_container_width=True)

# Main content
if analyze_button:
    # Prepare data
    driver_data = {
        "Speed": float(speed),
        "Alertness": float(alertness),
        "Seatbelt": int(seatbelt),
        "HR": float(hr),
        "Fatigue": int(fatigue),
        "speed_change": float(speed_change),
        "prev_alertness": float(prev_alertness)
    }
    
    # Show loading
    with st.spinner("Analyzing driver condition..."):
        try:
            response = requests.post(
                f"{API_URL}/v1/analyze",
                json=driver_data,
                headers=HEADERS,
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Top metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    prediction = result['ml_prediction']
                    color = "🔴" if prediction == "Drowsy" else "🟢"
                    st.metric("Prediction", f"{color} {prediction}")
                
                with col2:
                    confidence = result['ml_confidence'] * 100
                    st.metric("Confidence", f"{confidence:.1f}%", 
                             delta=result['confidence_level'])
                
                with col3:
                    risk_score = result['risk_score']
                    risk_color = "🔴" if risk_score >= 70 else "🟡" if risk_score >= 40 else "🟢"
                    st.metric("Risk Score", f"{risk_color} {risk_score}/100")
                
                with col4:
                    st.metric("Risk State", result['risk_state'])
                
                st.divider()
                
                # Decision and Risk Factors
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("💡 Recommended Action")
                    decision = result['decision']
                    
                    if decision['severity'] == 'HIGH':
                        st.error(f"**{decision['action']}**")
                    elif decision['severity'] == 'MEDIUM':
                        st.warning(f"**{decision['action']}**")
                    else:
                        st.success(f"**{decision['action']}**")
                    
                    st.info(f"ℹ️ {decision['message']}")
                
                with col2:
                    st.subheader("⚠️ Risk Factors")
                    if result['risk_factors']:
                        for factor in result['risk_factors']:
                            st.markdown(f"- 🔸 {factor}")
                    else:
                        st.success("✅ No risk factors detected")
                
                st.divider()
                
                # Risk Score Gauge
                st.subheader("📊 Risk Assessment Visualization")
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Score", 'font': {'size': 24}},
                    delta={'reference': 40, 'increasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 40], 'color': '#90EE90'},
                            {'range': [40, 70], 'color': '#FFD700'},
                            {'range': [70, 100], 'color': '#FF6B6B'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature Importance
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔍 Top Contributing Features")
                    if result['top_contributing_features']:
                        for feat in result['top_contributing_features']:
                            st.markdown(f"**{feat['feature']}**: {feat['feature_value']:.2f}")
                            st.progress(float(feat['global_importance']))
                
                with col2:
                    st.subheader("📝 Explanations")
                    for exp in result['explanations'][:5]:
                        st.markdown(f"- {exp}")
                
                # Performance metrics
                st.divider()
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Model Version", result['model_version'])
                with col2:
                    st.metric("Model Type", result['model_type'])
                with col3:
                    st.metric("Inference Time", f"{result['inference_latency_ms']:.2f}ms")
                
            else:
                st.error(f"❌ API Error: {response.status_code}")
                st.json(response.json())
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure it's running:")
            st.code("uvicorn src.app:app --reload")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

else:
    # Welcome screen
    st.info("👈 Adjust driver parameters in the sidebar and click **Analyze Driver** to get predictions")
    
    # Show example scenarios
    st.subheader("📋 Example Scenarios")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("**✅ Alert Driver**")
        st.markdown("""
        - Speed: 80 km/h
        - Alertness: 90%
        - Fatigue: 2/10
        - HR: 72 bpm
        """)
    
    with col2:
        st.warning("**⚠️ Moderate Risk**")
        st.markdown("""
        - Speed: 65 km/h
        - Alertness: 55%
        - Fatigue: 5/10
        - HR: 85 bpm
        """)
    
    with col3:
        st.error("**🔴 High Risk**")
        st.markdown("""
        - Speed: 45 km/h
        - Alertness: 30%
        - Fatigue: 8/10
        - HR: 58 bpm
        """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🚗 Driver Drowsiness Detection System | Production ML System</p>
    <p>Built with FastAPI + scikit-learn + Streamlit</p>
</div>
""", unsafe_allow_html=True)
