import streamlit as st
import requests
import pandas as pd
import json
import os

st.set_page_config(page_title="SOC AI Operations", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for SOC UI
st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    h1, h2, h3 {color: #00FFCC;}
    .stAlert {background-color: rgba(255, 75, 75, 0.1); border: 1px solid #FF4B4B;}
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Alpha SecOps: AI Intelligence Platform")
st.markdown("Monitor, analyze, and interpret network traffic using **XGBoost & Groq LLaMA**.")

API_URL = os.getenv("API_URL", "http://backend:8000")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Single Analyst View", "📂 Batch Threat Hunting", "🧠 Model Intelligence"])

with tab1:
    st.subheader("Manual Threat Analysis")
    st.markdown("Enter comma-separated values for the 35 raw features. (Standard UNSW-NB15 format)")
    
    # Default values to make recruiter demo easier
    default_vals = "0.121478,tcp,ftp,FIN,6,8,258,350,111.954434,14158.94238,20689.83203,1,2,24.295601,17.354,14.659851,19.348616,255,2367469771,988892225,255,0.063,0.024,0.039,43,44,1,0,1,1,1,1,0,0,Normal"
    feature_input = st.text_area("Flow Features (35 comma-separated)", value=default_vals, height=100)
    
    if st.button("🚀 Analyze Single Flow", type="primary"):
        with st.spinner("Analyzing with AI Model..."):
            try:
                values = [x.strip() for x in feature_input.split(",")]
                if len(values) != 35:
                    st.error("❌ Exactly 35 values required.")
                else:
                    data = {
                        "dur": float(values[0]), "proto": values[1], "service": values[2], "state": values[3],
                        "spkts": float(values[4]), "dpkts": float(values[5]), "sbytes": float(values[6]), "dbytes": float(values[7]),
                        "rate": float(values[8]), "sload": float(values[9]), "dload": float(values[10]), "sloss": float(values[11]),
                        "dloss": float(values[12]), "sinpkt": float(values[13]), "dinpkt": float(values[14]), "sjit": float(values[15]),
                        "djit": float(values[16]), "swin": float(values[17]), "stcpb": float(values[18]), "dtcpb": float(values[19]),
                        "dwin": float(values[20]), "tcprtt": float(values[21]), "synack": float(values[22]), "ackdat": float(values[23]),
                        "smean": float(values[24]), "dmean": float(values[25]), "trans_depth": float(values[26]), "response_body_len": float(values[27]),
                        "ct_src_dport_ltm": float(values[28]), "ct_dst_sport_ltm": float(values[29]), "is_ftp_login": float(values[30]),
                        "ct_ftp_cmd": float(values[31]), "ct_flw_http_mthd": float(values[32]), "is_sm_ips_ports": float(values[33]),
                        "attack_cat": values[34]
                    }

                    response = requests.post(f"{API_URL}/analyze", json=data)
                    
                    if response.status_code == 200:
                        res = response.json()
                        st.divider()
                        
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            if res["prediction"] == 1:
                                st.error("🚨 THREAT DETECTED")
                                st.metric("Decision", "ATTACK")
                                
                                st.markdown("### 🔍 Model Explainability (SHAP Top Features)")
                                top_feats = res.get("top_features", {})
                                if top_feats:
                                    feat_df = pd.DataFrame(list(top_feats.items()), columns=["Feature", "Impact"]).set_index("Feature")
                                    st.bar_chart(feat_df)
                            else:
                                st.success("✅ NORMAL TRAFFIC")
                                st.metric("Decision", "NORMAL")
                                
                        with col2:
                            st.markdown("### 🤖 CTI Copilot (LLaMA3)")
                            st.info(res["explanation"])
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error processing input: {e}")

with tab2:
    st.subheader("Batch CSV Threat Hunting")
    uploaded_file = st.file_uploader("Upload Network PCAP/Flow CSV", type=["csv"])
    
    if uploaded_file is not None:
        if st.button("🔍 Scan File"):
            with st.spinner("Processing batch file through XGBoost engine..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                res = requests.post(f"{API_URL}/analyze-csv", files=files)
                
                if res.status_code == 200:
                    results = res.json()["results"]
                    df_res = pd.DataFrame(results)
                    
                    attacks = df_res[df_res["prediction"] == 1].shape[0]
                    total = df_res.shape[0]
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Flows Scanned", total)
                    col2.metric("Threats Detected", attacks, delta=attacks, delta_color="inverse")
                    col3.metric("Normal Flows", total - attacks)
                    
                    st.dataframe(df_res.style.applymap(lambda x: "background-color: #FF4B4B" if x == "Attack" else "", subset=["label"]), use_container_width=True)
                else:
                    st.error("API Error during batch analysis.")

with tab3:
    st.subheader("Model Architectures & Evaluation")
    st.markdown("Review the tracked experiments from the latest `train_pipeline.py` run via MLFlow/JSON tracking.")
    
    import glob
    tracking_dir = "/app/mlflow_tracking"
    if not os.path.exists(tracking_dir) and os.path.exists("../mlflow_tracking"):
        tracking_dir = "../mlflow_tracking"
        
    try:
        files = glob.glob(os.path.join(tracking_dir, "*.json"))
        if files:
            all_metrics = []
            for f in files:
                with open(f, "r") as json_f:
                    data = json.load(json_f)
                    row = {"Model": data["model_name"], "Timestamp": data["timestamp"]}
                    row.update(data["metrics"])
                    all_metrics.append(row)
                    
            metrics_df = pd.DataFrame(all_metrics).sort_values("f1_score", ascending=False)
            st.dataframe(metrics_df, use_container_width=True)
            
            st.markdown("### Feature Matrix Schema:")
            st.json({"Pipeline Structure": ["Raw Data", "Engineered Ratios", "RobustScaling", "OneHotEncoding", "XGBoost Classifier"]})
        else:
            st.info("No experiment logs found. Run `python training/train_pipeline.py` to populate.")
    except Exception as e:
        st.warning("Could not load tracking metrics.")
