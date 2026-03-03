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

def generate_explanation(prediction: int):
    """
    Generates AI explanation for IDS prediction
    """

    context = ATTACK_CONTEXT.get(prediction, "Unknown network behavior detected.")

    prompt = f"""
You are an expert cybersecurity analyst.

A network intrusion detection system flagged traffic as:
{context}

Explain:
1. What this means in simple terms
2. Why it could be dangerous
3. Real-world impact
4. How to mitigate it

Keep it clear, concise, and professional.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # fast + powerful
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content