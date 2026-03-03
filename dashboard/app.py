import streamlit as st
import requests

st.set_page_config(page_title="AI Security Intelligence", layout="wide")

st.title("🛡️ AI Security Intelligence Platform")

st.markdown("Detect and analyze network threats using AI + LLM reasoning.")

API_URL =  "http://backend:8000/analyze"

st.divider()

st.subheader("🔢 Enter Network Features")

feature_input = st.text_area(
    "Paste comma-separated feature values (same order as model)",
    height=150
)

if st.button("🚀 Analyze Traffic"):

    try:
        features = [float(x.strip()) for x in feature_input.split(",")]

        response = requests.post(API_URL, json={"features": features})

        if response.status_code == 200:
            result = response.json()

            st.success("Analysis Complete!")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Prediction", result["label"])
                st.metric("Raw Value", result["prediction"])

            with col2:
                st.markdown("### 🧠 AI Explanation")
                st.write(result["explanation"])

        else:
            st.error("API Error. Check FastAPI server.")

    except Exception as e:
        st.error(f"Error: {e}")