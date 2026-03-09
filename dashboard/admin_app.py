"""
Driver Drowsiness Detection - Complete Admin Dashboard
=======================================================
Full-featured UI with all API endpoints and monitoring
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time

# Configuration
API_URL = "http://localhost:8000"
API_KEY = "dev_secure_key_123"
HEADERS = {"x-api-key": API_KEY}

# Page config
st.set_page_config(
    page_title="Driver Safety - Admin Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("🚗 Driver Drowsiness Detection - Admin Dashboard")
st.markdown("### Complete Production ML System Monitoring")

# Sidebar - Navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🎯 Driver Analysis", "📈 System Metrics", "🔍 Drift Detection", 
     "⚙️ Model Management", "📋 Audit Logs", "🧪 Batch Processing"]
)

# ============================================================================
# PAGE 1: Driver Analysis
# ============================================================================
if page == "🎯 Driver Analysis":
    st.header("🎯 Real-Time Driver Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Driver Inputs")
        speed = st.slider("Speed (km/h)", 0, 200, 60)
        alertness = st.slider("Alertness", 0.0, 1.0, 0.8, 0.01)
        seatbelt = st.selectbox("Seatbelt", [1, 0], format_func=lambda x: "On" if x == 1 else "Off")
        hr = st.slider("Heart Rate (bpm)", 30, 200, 75)
        fatigue = st.slider("Fatigue Level", 0, 10, 3)
        speed_change = st.slider("Speed Variability", 0.0, 20.0, 5.0)
        prev_alertness = st.slider("Previous Alertness", 0.0, 1.0, 0.85, 0.01)
        
        analyze_btn = st.button("🔍 Analyze Driver", type="primary", use_container_width=True)
    
    with col2:
        if analyze_btn:
            driver_data = {
                "Speed": float(speed),
                "Alertness": float(alertness),
                "Seatbelt": int(seatbelt),
                "HR": float(hr),
                "Fatigue": int(fatigue),
                "speed_change": float(speed_change),
                "prev_alertness": float(prev_alertness)
            }
            
            try:
                response = requests.post(f"{API_URL}/v1/analyze", json=driver_data, headers=HEADERS, timeout=5)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Prediction", result['ml_prediction'])
                    m2.metric("Confidence", f"{result['ml_confidence']*100:.1f}%")
                    m3.metric("Risk Score", f"{result['risk_score']}/100")
                    m4.metric("Risk State", result['risk_state'])
                    
                    # Risk Gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result['risk_score'],
                        title={'text': "Risk Score"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'steps': [
                                {'range': [0, 40], 'color': "lightgreen"},
                                {'range': [40, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {'line': {'color': "red", 'width': 4}, 'value': 70}
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Details
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.subheader("💡 Decision")
                        st.info(f"**{result['decision']['action']}**\n\n{result['decision']['message']}")
                        
                        st.subheader("⚠️ Risk Factors")
                        if result['risk_factors']:
                            for factor in result['risk_factors']:
                                st.markdown(f"- {factor}")
                        else:
                            st.success("No risk factors")
                    
                    with col_b:
                        st.subheader("🔍 Top Features")
                        for feat in result['top_contributing_features'][:5]:
                            st.markdown(f"**{feat['feature']}**: {feat['feature_value']:.2f}")
                            st.progress(float(feat['global_importance']))
                        
                        st.subheader("📝 Explanations")
                        for exp in result['explanations'][:5]:
                            st.markdown(f"- {exp}")
                    
                    # Performance
                    st.divider()
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Model Version", result['model_version'])
                    p2.metric("Model Type", result['model_type'])
                    p3.metric("Latency", f"{result['inference_latency_ms']:.2f}ms")
                    
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")

# ============================================================================
# PAGE 2: System Metrics
# ============================================================================
elif page == "📈 System Metrics":
    st.header("📈 System Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("System Metrics")
        try:
            metrics = requests.get(f"{API_URL}/v1/metrics").json()
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Requests", metrics['total_requests'])
            m2.metric("Drowsy Predictions", metrics['drowsy_predictions'])
            m3.metric("Avg Latency", f"{metrics['average_latency_ms']:.2f}ms")
            
            # Prediction distribution
            if metrics['total_requests'] > 0:
                alert_count = metrics['total_requests'] - metrics['drowsy_predictions']
                fig = go.Figure(data=[go.Pie(
                    labels=['Alert', 'Drowsy'],
                    values=[alert_count, metrics['drowsy_predictions']],
                    hole=.3
                )])
                fig.update_layout(title="Prediction Distribution")
                st.plotly_chart(fig, use_container_width=True)
        except:
            st.error("Cannot fetch metrics")
    
    with col2:
        st.subheader("Model Performance")
        try:
            perf = requests.get(f"{API_URL}/v1/model/performance", headers=HEADERS).json()
            
            p1, p2 = st.columns(2)
            p1.metric("Alert Predictions", perf['prediction_distribution']['Alert'])
            p2.metric("Drowsy Predictions", perf['prediction_distribution']['Drowsy'])
            
            st.metric("Average Confidence", f"{perf['avg_confidence']:.3f}")
            st.metric("Total Predictions", perf['total_predictions'])
            
            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=perf['avg_confidence'],
                title={'text': "Average Confidence"},
                gauge={'axis': {'range': [0, 1]}}
            ))
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.error("Cannot fetch performance metrics")
    
    # Diagnostics
    st.divider()
    st.subheader("🔧 System Diagnostics")
    try:
        diag = requests.get(f"{API_URL}/v1/diagnostics").json()
        
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Model Loaded", "✅" if diag['model_loaded'] else "❌")
        d2.metric("Total Requests", diag['total_requests'])
        d3.metric("Drowsy Ratio", f"{diag['drowsy_ratio']:.2%}")
        d4.metric("Model Version", diag['model_version'])
        
        st.json(diag)
    except:
        st.error("Cannot fetch diagnostics")

# ============================================================================
# PAGE 3: Drift Detection
# ============================================================================
elif page == "🔍 Drift Detection":
    st.header("🔍 Data Drift Detection")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Actions")
        
        if st.button("📊 Set Baseline", use_container_width=True):
            try:
                response = requests.post(f"{API_URL}/v1/drift/baseline", headers=HEADERS)
                if response.status_code == 200:
                    st.success("✅ Baseline set successfully")
                else:
                    st.error(f"Error: {response.json()}")
            except Exception as e:
                st.error(f"Error: {e}")
        
        if st.button("🔍 Detect Drift", use_container_width=True):
            try:
                response = requests.get(f"{API_URL}/v1/drift/detect", headers=HEADERS)
                drift = response.json()
                
                st.subheader("Drift Status")
                if drift['status'] == "Stable":
                    st.success(f"✅ {drift['status']}")
                elif drift['status'] == "Drift Warning":
                    st.warning(f"⚠️ {drift['status']}")
                else:
                    st.info(drift['status'])
                
                if drift.get('details'):
                    st.subheader("Drift Details")
                    st.json(drift['details'])
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        st.subheader("Drift Monitoring")
        st.info("""
        **Drift Detection Monitors:**
        - Feature mean shifts (>20%)
        - Feature variance shifts (>20%)
        - Prediction distribution changes (>15%)
        
        **Buffer Size:** 1000 samples (rolling window)
        
        **Actions:**
        1. Set baseline after collecting initial data
        2. Monitor drift regularly
        3. Retrain model if drift detected
        """)
        
        st.subheader("How It Works")
        st.markdown("""
        1. **Collect Data**: System tracks last 1000 predictions
        2. **Set Baseline**: Capture current distribution
        3. **Monitor**: Compare live data vs baseline
        4. **Alert**: Warn when significant drift detected
        """)

# ============================================================================
# PAGE 4: Model Management
# ============================================================================
elif page == "⚙️ Model Management":
    st.header("⚙️ Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Switch Model Version")
        version = st.text_input("Model Version", "v2.0")
        
        if st.button("🔄 Switch Model", use_container_width=True):
            try:
                response = requests.post(f"{API_URL}/v1/model/switch/{version}", headers=HEADERS)
                if response.status_code == 200:
                    st.success(f"✅ {response.json()['message']}")
                else:
                    st.error(f"❌ {response.json()['detail']}")
            except Exception as e:
                st.error(f"Error: {e}")
        
        st.info("⚠️ Model switching requires versioned models in `models/{version}/` directory")
    
    with col2:
        st.subheader("Shadow Model Testing")
        shadow_version = st.text_input("Shadow Version", "v2.0")
        
        if st.button("🔬 Load Shadow Model", use_container_width=True):
            try:
                response = requests.post(f"{API_URL}/v1/model/shadow/{shadow_version}", headers=HEADERS)
                if response.status_code == 200:
                    st.success(f"✅ {response.json()['message']}")
                else:
                    st.error(f"❌ {response.json()['detail']}")
            except Exception as e:
                st.error(f"Error: {e}")
        
        if st.button("❌ Disable Shadow Model", use_container_width=True):
            try:
                response = requests.post(f"{API_URL}/v1/model/shadow/disable", headers=HEADERS)
                st.success("✅ Shadow model disabled")
            except Exception as e:
                st.error(f"Error: {e}")
        
        st.info("💡 Shadow models run in parallel for A/B testing without affecting production")

# ============================================================================
# PAGE 5: Audit Logs
# ============================================================================
elif page == "📋 Audit Logs":
    st.header("📋 Audit Logs")
    
    limit = st.slider("Number of logs", 5, 50, 10)
    
    if st.button("🔄 Refresh Logs", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/v1/audit/recent?limit={limit}")
            logs = response.json()
            
            if logs.get('records'):
                st.success(f"✅ Showing {logs['count']} recent logs")
                
                for i, record in enumerate(logs['records']):
                    with st.expander(f"Log {i+1} - {record.get('trace_id', 'N/A')}"):
                        st.json(record)
            else:
                st.info("No audit logs available yet")
        except Exception as e:
            st.error(f"Error: {e}")

# ============================================================================
# PAGE 6: Batch Processing
# ============================================================================
elif page == "🧪 Batch Processing":
    st.header("🧪 Batch Processing")
    
    st.subheader("Upload Batch Data")
    
    # Sample data generator
    if st.button("Generate Sample Batch (10 items)"):
        import random
        batch = []
        for _ in range(10):
            batch.append({
                "Speed": random.uniform(30, 120),
                "Alertness": random.uniform(0.3, 1.0),
                "Seatbelt": random.choice([0, 1]),
                "HR": random.uniform(50, 120),
                "Fatigue": random.randint(0, 10),
                "speed_change": random.uniform(0, 20),
                "prev_alertness": random.uniform(0.3, 1.0)
            })
        
        st.session_state['batch_data'] = batch
        st.success("✅ Sample batch generated")
    
    if 'batch_data' in st.session_state:
        st.dataframe(pd.DataFrame(st.session_state['batch_data']))
        
        if st.button("🚀 Process Batch", type="primary"):
            try:
                response = requests.post(
                    f"{API_URL}/v1/analyze/batch",
                    json=st.session_state['batch_data'],
                    headers=HEADERS
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ Processed {result['total_processed']} items")
                    
                    # Results summary
                    results_df = pd.DataFrame(result['results'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Alert", len(results_df[results_df['ml_prediction'] == 'Alert']))
                    with col2:
                        st.metric("Drowsy", len(results_df[results_df['ml_prediction'] == 'Drowsy']))
                    
                    # Show results
                    st.dataframe(results_df[['ml_prediction', 'ml_confidence', 'risk_score', 'risk_state']])
                else:
                    st.error(f"Error: {response.json()}")
            except Exception as e:
                st.error(f"Error: {e}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🚗 Driver Drowsiness Detection System | Production ML System</p>
    <p>Complete Admin Dashboard with All API Endpoints</p>
</div>
""", unsafe_allow_html=True)
