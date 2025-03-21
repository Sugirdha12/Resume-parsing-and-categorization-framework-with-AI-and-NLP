import streamlit as st
import pdfplumber
import requests
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ Hardcoded Cohere API Key
COHERE_API_KEY = "KfRekS4l6ZtIsjIhiu8SMjAVnz3OXsMz5X5S6GP2"

# ✅ Load ML Models
clf = pickle.load(open('clf_rf1.pkl', 'rb'))  # Random Forest Model
tfidf = pickle.load(open('tfidf1.pkl', 'rb'))  # TF-IDF Vectorizer

# ✅ Category Mapping
category_mapping = {
    15: "Java Developer", 23: "Testing", 8: "DevOps Engineer",
    20: "Python Developer", 24: "Web Designing", 12: "HR",
    13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
    18: "Operations Manager", 6: "Data Science", 22: "Sales",
    16: "Mechanical Engineer", 1: "Arts", 7: "Database",
    11: "Electrical Engineering", 14: "Health and Fitness",
    19: "PMO", 4: "Business Analyst", 9: "DotNet Developer",
    2: "Automation Testing", 17: "Network Security Engineer",
    21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate"
}

# ✅ Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# ✅ Function to calculate ATS Score
def calculate_ats_score(job_description, resume_text):
    job_keywords = set(re.findall(r'\b\w+\b', job_description.lower()))
    resume_keywords = set(re.findall(r'\b\w+\b', resume_text.lower()))
    matched_keywords = job_keywords.intersection(resume_keywords)
    ats_score = len(matched_keywords) / len(job_keywords) * 100 if job_keywords else 0
    return round(ats_score, 2)

# ✅ Function to improve resume using Cohere AI
def improve_resume_with_ai(api_key, resume_text, job_description):
    url = "https://api.cohere.ai/chat"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    query = f"""
    The following is a job description and a resume. Modify the resume to align it with the job description, 
    ensuring it highlights the necessary skills, experiences, and keywords to achieve a higher ATS score.

    *Job Description:*
    {job_description}

    *Resume:*
    {resume_text}

    *Modified Resume:*
    """
    
    payload = {
        "query": query,
        "temperature": 0.7,
        "max_tokens": 1000
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        modified_resume = response_json.get("text", "").strip()

        if not modified_resume:
            return "⚠ Error: No response from Cohere AI. Try again later."

        return modified_resume

    except requests.exceptions.RequestException as e:
        return f"⚠ API Request Error: {str(e)}"

# ✅ Function to ask AI a question about the resume
def ask_question(question, resume_text):
    url = "https://api.cohere.ai/chat"
    headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "query": f"Resume Content:\n{resume_text}\n\nQuestion: {question}",
        "temperature": 0.1,
        "max_tokens": 500
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        return response_json.get('text', "Sorry, I couldn't find the answer.").strip()
    except requests.exceptions.RequestException as e:
        return f"⚠ API Request Error: {str(e)}"

# ✅ Function to predict resume category
def predict_resume_category(resume_text):
    input_features = tfidf.transform([resume_text])
    prediction_probabilities = clf.predict_proba(input_features)
    prediction_id = np.argmax(prediction_probabilities)
    predicted_category = category_mapping.get(prediction_id, "Unknown")
    return predicted_category

# ✅ Apply Modern Styling for Light Theme
st.markdown("""
    <style>
        body {
            background-color: #ffffff;
            color: #333;
            font-family: 'Arial', sans-serif;
        }
        .stTitle {
            color: #007bff;
            font-size: 24px;
            font-weight: bold;
        }
        .stButton > button {
            background: linear-gradient(45deg, #007bff, #6610f2);
            color: white;
            border-radius: 12px;
            padding: 12px;
            font-weight: bold;
            font-size: 16px;
        }
        .stTextArea, .stTextInput, .stFileUploader {
            border-radius: 12px;
            border: 1px solid #007bff;
            padding: 10px;
        }
        .stSuccess {
            background-color: #d4edda;
            border-radius: 12px;
            padding: 12px;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Streamlit Web App Title
st.title("🚀 AI Resume Analyzer")
st.write("Enhance your resume with AI, boost your ATS score, and get career insights!")

# ✅ Upload PDF Resume
uploaded_file = st.file_uploader("📤 Upload Resume PDF", type="pdf")

if uploaded_file:
    st.subheader(f"📄 Processing: {uploaded_file.name}")

    # ✅ Extract Text
    resume_text = extract_text_from_pdf(uploaded_file)
    st.text_area("📜 Extracted Resume Text:", resume_text, height=200)

    # ✅ ATS Score Calculation
    job_description = st.text_area("📝 Enter Job Description for ATS Score:")
    if st.button("📊 Calculate ATS Score"):
        ats_score = calculate_ats_score(job_description, resume_text)
        st.success(f"✅ ATS Score: {ats_score}%")

    # ✅ Improve Resume with AI
    if st.button("✨ Improve Resume for Higher ATS Score"):
        modified_resume = improve_resume_with_ai(COHERE_API_KEY, resume_text, job_description)
        st.text_area("📌 AI-Optimized Resume:", modified_resume, height=200)

    # ✅ Ask AI Questions About Resume
    question = st.text_input("❓ Ask a question about the resume:")
    if st.button("🤖 Get AI Response"):
        answer = ask_question(question, resume_text)
        st.success(f"🧠 AI Response: {answer}")

    # ✅ Resume Classification
    if st.button("🎯 Classify Resume into Job Category"):
        predicted_category = predict_resume_category(resume_text)
        st.success(f"🏆 Resume Classified as: {predicted_category}")