import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_batch_insights(stats: dict) -> list:
    """
    Generates AI business insights based on the batch statistics using Groq LLaMA 3.
    """
    prompt = f"""
You are a Lead Data Analyst reviewing a batch of network traffic statistics from an Intrusion Detection System.

Here is the JSON summary of the latest batch scan:
{stats}

Your task:
Analyze these statistics and generate exactly 3 concise, business-focused insights highlighting the most alarming trends, anomalies, or important data points. 
Format your response as a simple markdown bulleted list. 
Do not include introductory or concluding text. Just the 3 bullet points.
Make them sound professional, actionable, and analytical.
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        
        content = response.choices[0].message.content
        bullets = [line.strip() for line in content.split('\n') if line.strip().startswith('-') or line.strip().startswith('*')]
        
        if not bullets:
            bullets = [line.strip() for line in content.split('\n') if line.strip()]
            
        return bullets[:3]
    except Exception as e:
        print(f"Error generating insights: {e}")
        return [
            "- Could not generate AI insights at this time.", 
            "- Check API key configuration. Data is still available in the visual dashboard.",
            "- Please review manual statistics."
        ]
