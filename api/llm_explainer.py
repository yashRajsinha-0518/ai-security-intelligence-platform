import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ===============================
# Cybersecurity Knowledge Base
# ===============================
ATTACK_CONTEXT = {
    0: "Normal network traffic with no malicious indicators.",
    1: "Potential malicious activity detected based on network telemetry patterns."
}

def generate_explanation(prediction: int, top_features_dict: dict = None):
    """
    Generates AI explanation for IDS prediction using SHAP feature importances.
    """

    context = ATTACK_CONTEXT.get(prediction, "Unknown network behavior detected.")
    
    features_desc = ""
    if top_features_dict:
        features_desc = "\\nTop Features Driving This Decision:\\n"
        for i, (k, v) in enumerate(top_features_dict.items()):
            features_desc += f"{i+1}. {k} (Impact score: {v:.4f})\\n"

    prompt = f"""
You are an expert cybersecurity analyst.

A network intrusion detection system flagged traffic as:
{context}
{features_desc}

Explain:
1. What this means in simple terms.
2. Why these specific features (if provided) could indicate malicious behavior.
3. Real-world impact.
4. How to mitigate it.

Keep it highly professional, structured, and insightful for a Security Operations Center (SOC) dashboard. Use Markdown.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content