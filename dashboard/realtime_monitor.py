"""
MAANG-Level Real-Time Driver Monitoring Dashboard
Simulates live sensor data streaming from vehicle telemetry
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

API_URL = "http://localhost:8000"
API_KEY = "dev_secure_key_123"
HEADERS = {"x-api-key": API_KEY}

st.set_page_config(page_title="Fleet Monitor", page_icon="🚗", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .big-font {font-size:50px !important; font-weight:bold; text-align:center;}
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">🚗 Real-Time Fleet Monitoring System</p>', unsafe_allow_html=True)
st.markdown("### Live Telemetry from 1,247 Active Vehicles")

# Initialize session state
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'history' not in st.session_state:
    st.session_state.history = []

# Simulate realistic driving data
def generate_realistic_data(scenario="normal"):
    base_time = datetime.now()
    
    if scenario == "normal":
        return {
            "Speed": np.random.uniform(60, 90),
            "Alertness": np.random.uniform(0.75, 0.95),
            "Seatbelt": 1,
            "HR": np.random.uniform(65, 80),
            "Fatigue": np.random.randint(1, 4),
            "speed_change": np.random.uniform(2, 8),
            "prev_alertness": np.random.uniform(0.75, 0.95)
        }
    elif scenario == "tired":
        return {
            "Speed": np.random.uniform(50, 70),
            "Alertness": np.random.uniform(0.4, 0.6),
            "Seatbelt": 1,
            "HR": np.random.uniform(60, 75),
            "Fatigue": np.random.randint(6, 8),
            "speed_change": np.random.uniform(1, 5),
            "prev_alertness": np.random.uniform(0.6, 0.8)
        }
    else:  # critical
        return {
            "Speed": np.random.uniform(30, 50),
            "Alertness": np.random.uniform(0.2, 0.4),
            "Seatbelt": 1,
            "HR": np.random.uniform(55, 65),
            "Fatigue": np.random.randint(8, 10),
            "speed_change": np.random.uniform(0, 3),
            "prev_alertness": np.random.uniform(0.5, 0.7)
        }

# Top metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card"><h2>1,247</h2><p>Active Vehicles</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><h2>23</h2><p>Alerts Today</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><h2>99.7%</h2><p>System Uptime</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><h2>12ms</h2><p>Avg Latency</p></div>', unsafe_allow_html=True)

st.divider()

# Control panel
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.subheader("🎯 Live Vehicle Monitoring")
    vehicle_id = st.selectbox("Select Vehicle", ["VEH-2847", "VEH-1923", "VEH-4521"])

with col2:
    scenario = st.selectbox("Simulation Mode", ["normal", "tired", "critical"])

with col3:
    st.write("")
    st.write("")
    if st.button("▶️ Start Monitoring" if not st.session_state.monitoring else "⏸️ Stop Monitoring", 
                 type="primary", use_container_width=True):
        st.session_state.monitoring = not st.session_state.monitoring

# Real-time monitoring
if st.session_state.monitoring:
    placeholder = st.empty()
    
    for i in range(100):  # Simulate 100 data points
        data = generate_realistic_data(scenario)
        
        try:
            resp = requests.post(f"{API_URL}/v1/analyze", json=data, headers=HEADERS, timeout=2)
            
            if resp.status_code == 200:
                result = resp.json()
                
                # Store history
                st.session_state.history.append({
                    'timestamp': datetime.now(),
                    'prediction': result['ml_prediction'],
                    'confidence': result['ml_confidence'],
                    'risk_score': result['risk_score']
                })
                
                # Keep only last 50 points
                if len(st.session_state.history) > 50:
                    st.session_state.history.pop(0)
                
                with placeholder.container():
                    # Current status
                    col1, col2, col3, col4 = st.columns(4)
                    
                    status_color = "🔴" if result['ml_prediction'] == "Drowsy" else "🟢"
                    col1.metric("Status", f"{status_color} {result['ml_prediction']}")
                    col2.metric("Confidence", f"{result['ml_confidence']*100:.1f}%", 
                               delta=result['confidence_level'])
                    col3.metric("Risk Score", f"{result['risk_score']}/100")
                    col4.metric("Risk State", result['risk_state'])
                    
                    # Alert banner
                    if result['risk_score'] >= 70:
                        st.error(f"🚨 **CRITICAL ALERT**: {result['decision']['action']}")
                    elif result['risk_score'] >= 40:
                        st.warning(f"⚠️ **WARNING**: {result['decision']['action']}")
                    else:
                        st.success(f"✅ **NORMAL**: {result['decision']['message']}")
                    
                    # Real-time charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Risk score over time
                        if len(st.session_state.history) > 1:
                            df = pd.DataFrame(st.session_state.history)
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=df['timestamp'],
                                y=df['risk_score'],
                                mode='lines+markers',
                                name='Risk Score',
                                line=dict(color='#667eea', width=3),
                                fill='tozeroy'
                            ))
                            fig.update_layout(
                                title="Risk Score Timeline",
                                height=300,
                                showlegend=False,
                                yaxis_range=[0, 100]
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Confidence gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=result['ml_confidence']*100,
                            title={'text': "Model Confidence"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#667eea"},
                                'steps': [
                                    {'range': [0, 55], 'color': "lightgray"},
                                    {'range': [55, 85], 'color': "lightblue"},
                                    {'range': [85, 100], 'color': "lightgreen"}
                                ]
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Telemetry data
                    st.subheader("📊 Live Telemetry")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Speed", f"{data['Speed']:.1f} km/h")
                    col2.metric("Heart Rate", f"{data['HR']:.0f} bpm")
                    col3.metric("Alertness", f"{data['Alertness']*100:.0f}%")
                    col4.metric("Fatigue", f"{data['Fatigue']}/10")
                    
                    # Risk factors
                    if result['risk_factors']:
                        st.warning("**Active Risk Factors:**")
                        for factor in result['risk_factors']:
                            st.write(f"• {factor}")
                    
                    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} | Latency: {result['inference_latency_ms']:.2f}ms")
        
        except Exception as e:
            st.error(f"Connection error: {e}")
            break
        
        time.sleep(2)  # Update every 2 seconds
        
        if not st.session_state.monitoring:
            break

else:
    st.info("👆 Click 'Start Monitoring' to begin real-time vehicle telemetry streaming")
    
    # Show sample dashboard
    st.subheader("📈 System Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Recent Alerts**")
        st.write("• VEH-2847: High fatigue detected - 2 min ago")
        st.write("• VEH-1923: Alertness drop - 15 min ago")
        st.write("• VEH-4521: Normal operation - 1 hour ago")
    
    with col2:
        st.markdown("**Fleet Statistics**")
        st.write("• Average confidence: 87.3%")
        st.write("• Total predictions today: 45,892")
        st.write("• Critical alerts: 23")

st.divider()
st.markdown("**🔗 API Endpoint:** `POST /v1/analyze` | **📊 Monitoring:** Real-time | **⚡ Latency:** <50ms")
