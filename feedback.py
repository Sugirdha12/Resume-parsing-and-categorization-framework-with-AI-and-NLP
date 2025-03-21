import os
import re
import requests
import pdfplumber

# Function to extract text from a PDF file using pdfplumber
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to extract specific sections from the resume: Education, Experience, Skills
def extract_sections_from_text(text):
    # Regular expression patterns to look for sections
    sections = {
        "education": r"(education|academic|qualification)(.*?)(experience|skills|$)",
        "experience": r"(experience|work history|professional experience)(.*?)(skills|education|$)",
        "skills": r"(skills|technical skills|technologies)(.*?)(experience|education|$)"
    }

    extracted_sections = {
        "education": "",
        "experience": "",
        "skills": ""
    }

    for section, pattern in sections.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            extracted_sections[section] = match.group(2).strip()
    
    return extracted_sections

# Function to analyze resume using Cohere API
def analyze_resume_with_cohere(resume_text, api_key):
    url = "https://api.cohere.ai/generate"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Constructing the prompt for Cohere to analyze the resume
    prompt = f"Please review the following resume content and provide feedback, focusing only on strengths, weaknesses, and areas for improvement.\n\nResume Content:\n{resume_text}"

    payload = {
        "model": "command-xlarge-nightly",
        "prompt": prompt,
        "max_tokens": 1000,
        "temperature": 0.5
    }

    response = requests.post(url, headers=headers, json=payload)
    response_json = response.json()

    try:
        feedback = response_json['text'].strip()
    except KeyError:
        feedback = "Sorry, I couldn't generate feedback."

    return feedback

# Main function
def main():
    pdf_path = input("Enter the path to the resume PDF file: ")

    # Validate if the file exists
    if not os.path.exists(pdf_path):
        print("The specified file does not exist. Please check the path and try again.")
        return

    # Validate if the file is a PDF
    if not pdf_path.lower().endswith('.pdf'):
        print("Invalid file format. Please enter the path of a PDF file.")
        return

    # Extract text from the PDF file
    resume_text = extract_text_from_pdf(pdf_path)

    # Extract key sections from the resume (education, experience, skills)
    sections = extract_sections_from_text(resume_text)

    # Analyze the entire resume using Cohere API
    api_key = '3K0BiE5jOql84dVFIPMwvA6pU8MVAL0deAmHMnyN'  # Replace with your Cohere API key
    feedback = analyze_resume_with_cohere(resume_text, api_key)

    # Print feedback from Cohere
    print("\nCohere's Feedback on the Resume (Strengths, Weaknesses, and Areas for Improvement):")
    print(feedback)

if __name__ == "__main__":
    main()
