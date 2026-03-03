🛡️ AI Security Intelligence Platform
# 🛡️ AI Security Intelligence Platform

> An end-to-end AI-powered Intrusion Detection System combining Machine Learning + LLM-based Cyber Threat Intelligence.

---

## 🚀 Overview

The **AI Security Intelligence Platform** is a full-stack cybersecurity system that:

- Detects malicious network traffic using a trained ML model
- Generates human-readable cyber threat explanations using LLM reasoning
- Exposes REST APIs via FastAPI
- Provides a real-time interactive dashboard using Streamlit

Unlike traditional IDS projects that stop at classification accuracy, this system focuses on **interpretability and actionable intelligence**.

---

## 🧠 System Architecture


Network Features → ML Model (XGBoost IDS)
→ Attack / Normal Prediction
→ LLM Reasoning Layer (Groq LLaMA)
→ Human-Readable Threat Intelligence
→ FastAPI Backend
→ Streamlit Dashboard


---

## 🏗️ Tech Stack

### 🔹 Machine Learning
- XGBoost Classifier
- Scikit-learn
- Pandas / NumPy
- UNSW-NB15 Dataset

### 🔹 AI Reasoning
- Groq API
- LLaMA 3 Model
- Structured Prompt Engineering

### 🔹 Backend
- FastAPI
- Uvicorn
- REST API
- OpenAPI (Swagger Docs)

### 🔹 Frontend
- Streamlit
- Interactive Dashboard UI

---

## 📦 Project Structure


AI-Security-Platform/
│
├── api/ # FastAPI backend + LLM integration
├── dashboard/ # Streamlit frontend
├── model/ # Trained model + preprocessing artifacts
├── data/ # UNSW dataset
├── utils/ # Helper modules
├── requirements.txt
├── .env.example
└── README.md


---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository


git clone https://github.com/yourusername/ai-security-intelligence-platform.git

cd ai-security-intelligence-platform


---

### 2️⃣ Create Virtual Environment


python -m venv venv
venv\Scripts\activate # Windows


---

### 3️⃣ Install Dependencies


pip install -r requirements.txt


---

### 4️⃣ Configure Environment Variables

Create a `.env` file:


GROQ_API_KEY=your_api_key_here


⚠️ Never upload your `.env` file to GitHub.

---

## ▶️ Running The Project

### 🔹 Start Backend


uvicorn api.main:app --reload


API Docs available at:


http://127.0.0.1:8000/docs


---

### 🔹 Start Dashboard

Open a new terminal:


streamlit run dashboard/app.py


Dashboard available at:


http://localhost:8501


---

## 🧪 API Endpoints

| Method | Endpoint | Description |
|--------|----------|------------|
| GET    | /        | Health check |
| POST   | /predict | Run ML prediction |
| GET    | /explain/{prediction} | Generate AI explanation |
| POST   | /analyze | Full ML + LLM pipeline |

---

## 🎯 Key Features

✔ Intrusion Detection using XGBoost  
✔ Mixed-type network feature preprocessing  
✔ Robust categorical encoding  
✔ Model artifact persistence  
✔ LLM-powered threat explanation  
✔ RESTful microservice architecture  
✔ Interactive visualization dashboard  

---

## 🧠 Design Philosophy

Traditional IDS systems answer:

> "Is this malicious?"

This platform answers:

> "Is this malicious, why, how dangerous is it, and what should be done?"

By combining statistical detection with LLM-based reasoning, the system bridges the gap between detection and actionable security intelligence.

---

## 📊 Dataset

- UNSW-NB15 Network Intrusion Dataset
- Binary classification: Normal vs Attack
- Real-world simulated network telemetry

---

## 🚀 Future Improvements

- Docker containerization
- Cloud deployment (Render / AWS)
- CSV upload support
- Real-time streaming detection
- Threat severity scoring
- Attack heatmap visualization

---

## 👨‍💻 Author

Yash Sinha  
AI / ML & Systems Engineering Enthusiast  

---

## 📜 License

MIT License
