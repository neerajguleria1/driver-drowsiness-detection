"""
Enhanced Driver Drowsiness Detection Dashboard
With all API features accessible
"""

import streamlit as st
import requests
import plotly.graph_objects as go

API_URL = "http://localhost:8000"
API_KEY = "dev_secure_key_123"
HEADERS = {"x-api-key": API_KEY}

st.set_page_config(page_title="Driver Safety Monitor", page_icon="🚗", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🚗 Driver Drowsiness Detection System</p>', unsafe_allow_html=True)

# Tabs for all features
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyze Driver", "📊 System Metrics", "🔧 Diagnostics", "📈 Performance"])

# TAB 1: Driver Analysis
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Driver Inputs")
        speed = st.slider("Speed (km/h)", 0, 200, 60)
        alertness = st.slider("Alertness", 0.0, 1.0, 0.8, 0.01)
        seatbelt = st.selectbox("Seatbelt", [1, 0], format_func=lambda x: "On" if x == 1 else "Off")
        hr = st.slider("Heart Rate", 30, 200, 75)
        fatigue = st.slider("Fatigue", 0, 10, 3)
        speed_change = st.slider("Speed Change", 0.0, 20.0, 5.0)
        prev_alertness = st.slider("Prev Alertness", 0.0, 1.0, 0.85, 0.01)
        
        analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    with col2:
        if analyze:
            data = {
                "Speed": float(speed),
                "Alertness": float(alertness),
                "Seatbelt": int(seatbelt),
                "HR": float(hr),
                "Fatigue": int(fatigue),
                "speed_change": float(speed_change),
                "prev_alertness": float(prev_alertness)
            }
            
            try:
                resp = requests.post(f"{API_URL}/v1/analyze", json=data, headers=HEADERS, timeout=5)
                if resp.status_code == 200:
                    result = resp.json()
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Prediction", result['ml_prediction'])
                    col2.metric("Confidence", f"{result['ml_confidence']*100:.1f}%")
                    col3.metric("Risk Score", f"{result['risk_score']}/100")
                    
                    st.success(f"**Action:** {result['decision']['action']}")
                    
                    if result['risk_factors']:
                        st.warning("**Risk Factors:**")
                        for factor in result['risk_factors']:
                            st.write(f"- {factor}")
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result['risk_score'],
                        title={'text': "Risk Score"},
                        gauge={'axis': {'range': [0, 100]},
                               'steps': [
                                   {'range': [0, 40], 'color': "lightgreen"},
                                   {'range': [40, 70], 'color': "yellow"},
                                   {'range': [70, 100], 'color': "red"}
                               ]}
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Error: {resp.status_code}")
            except Exception as e:
                st.error(f"Cannot connect to API: {e}")
        else:
            st.info("👈 Enter driver parameters and click Analyze")

# TAB 2: System Metrics
with tab2:
    if st.button("🔄 Refresh Metrics"):
        try:
            resp = requests.get(f"{API_URL}/v1/metrics")
            if resp.status_code == 200:
                metrics = resp.json()
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Requests", metrics['total_requests'])
                col2.metric("Drowsy Predictions", metrics['drowsy_predictions'])
                col3.metric("Avg Latency", f"{metrics['average_latency_ms']:.2f}ms")
            else:
                st.error("Failed to fetch metrics")
        except Exception as e:
            st.error(f"Error: {e}")

# TAB 3: Diagnostics
with tab3:
    if st.button("🔍 Run Diagnostics"):
        try:
            resp = requests.get(f"{API_URL}/v1/diagnostics")
            if resp.status_code == 200:
                diag = resp.json()
                st.json(diag)
            else:
                st.error("Failed to fetch diagnostics")
        except Exception as e:
            st.error(f"Error: {e}")

# TAB 4: Performance
with tab4:
    if st.button("📊 Get Performance"):
        try:
            resp = requests.get(f"{API_URL}/v1/model/performance", headers=HEADERS)
            if resp.status_code == 200:
                perf = resp.json()
                st.json(perf)
            else:
                st.error("Failed to fetch performance")
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.markdown("**🌐 API Documentation:** http://localhost:8000/docs")
