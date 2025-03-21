import re
import PyPDF2
import os
import requests

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ''
    try:
        with open(pdf_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

def calculate_ats_score(job_description, resume_text):
    """Calculates the ATS score by comparing the job description with the resume."""
    job_keywords = set(re.findall(r'\b\w+\b', job_description.lower()))
    resume_keywords = set(re.findall(r'\b\w+\b', resume_text.lower()))
    
    matched_keywords = job_keywords.intersection(resume_keywords)
    ats_score = len(matched_keywords) / len(job_keywords) * 100  # percentage
    
    return ats_score

def improve_resume_with_cohere(api_key, resume_text, job_description):
    """Improves the resume using Cohere Chat API."""
    url = "https://api.cohere.ai/chat"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Using simplified query instead of `messages` structure
    query = f"""
    The following is a job description and a resume. Modify the resume to align it with the job description, ensuring it highlights the necessary skills, experiences, and keywords to achieve a higher ATS score.

    Job Description:
    {job_description}

    Resume:
    {resume_text}

    Modified Resume:
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
        modified_resume = response_json.get("text", "").strip()  # Use `.get` to avoid KeyError
        return modified_resume
    except requests.exceptions.RequestException as e:
        print(f"Error with Cohere API request: {e}")
        return ""

# Main Functionality
def main():
    job_description = """We are seeking an innovative Generative AI Developer to design, develop, and deploy state-of-the-art generative models for applications like text generation, image synthesis, and AI-powered solutions. The ideal candidate will have expertise in machine learning, NLP, and frameworks like TensorFlow or PyTorch, with a passion for pushing the boundaries of AI technology."""

    resume_pdf_path = input("Enter the path to your resume PDF: ").strip()
    
    # Hardcoded API key
    api_key = "KfRekS4l6ZtIsjIhiu8SMjAVnz3OXsMz5X5S6GP2"

    if not os.path.exists(resume_pdf_path):
        print("Error: Resume PDF file does not exist.")
        return

    # Extract text from the resume
    resume_text = extract_text_from_pdf(resume_pdf_path)
    if not resume_text:
        print("Error: Could not extract text from the resume.")
        return

    # Calculate the initial ATS score
    ats_score = calculate_ats_score(job_description, resume_text)
    print(f"Original ATS Score: {ats_score:.2f}%")

    # Ask user if they are satisfied with the ATS score
    satisfaction = input("Are you satisfied with the ATS score? (yes/no): ").strip().lower()
    if satisfaction == "yes":
        print("Exiting the program.")
        return

    # Ask user if they want to modify the resume
    modify_choice = input("Should I modify your resume to improve the score? (yes/no): ").strip().lower()
    if modify_choice == "yes":
        modified_resume = improve_resume_with_cohere(api_key, resume_text, job_description)
        if not modified_resume:
            print("Error: Failed to generate a modified resume.")
            return

        print("\nModified Resume:")
        print(modified_resume)

        # Calculate the ATS score for the modified resume
        modified_ats_score = calculate_ats_score(job_description, modified_resume)
        print(f"Modified ATS Score: {modified_ats_score:.2f}%")
    else:
        print("Exiting the program.")

if __name__ == "__main__":
    main()
