import os
import sys
import re
import requests
import pdfplumber

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

# Function to extract text from all PDFs in a folder
def extract_text_from_pdfs_in_folder(folder_path):
    """Extracts text from all PDFs in a specified folder."""
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Extracting text from: {filename}")
            all_text += extract_text_from_pdf(pdf_path) + "\n\n"  # Add text from each PDF
    return all_text

# Function to ask a question using Cohere Chat API and get a response
def ask_question(question, context, api_key):
    """Sends a question to Cohere Chat API with the given context."""
    url = "https://api.cohere.ai/chat"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    query = f"""
    Here are the details from multiple resumes. Based on the information provided, answer the following question:

    Resume Details:
    {context}

    Question: {question}
    Answer:
    """
    payload = {
        "query": query,
        "temperature": 0.5,
        "max_tokens": 500
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise HTTPError for bad responses
        response_json = response.json()
        answer = response_json.get("text", "Sorry, I couldn't find the answer.").strip()
        return answer
    except requests.exceptions.RequestException as e:
        print(f"Error with Cohere API request: {e}")
        return "Sorry, there was an issue with the API request."

# Function to extract candidate names and contact numbers using regex
def extract_names_and_contacts(text):
    """Extracts candidate names and contact numbers using regex."""
    names = re.findall(r"[A-Za-z]+\s[A-Za-z]+", text)  # Match names (first and last)
    phone_numbers = re.findall(r"\+?\d{1,2}[-\s]?\(?\d{1,4}\)?[-\s]?\d{7,10}", text)  # Match phone numbers
    return names, phone_numbers

# Main function
def main():
    """Main function to interact with the user."""
    folder_path = input("Enter the folder path containing the PDF files: ")

    # Validate if the folder exists
    if not os.path.exists(folder_path):
        print("The specified folder does not exist. Please check the path and try again.")
        return

    # Extract text from all PDFs in the folder
    combined_text = extract_text_from_pdfs_in_folder(folder_path)

    # Your Cohere API key
    api_key = 'KfRekS4l6ZtIsjIhiu8SMjAVnz3OXsMz5X5S6GP2'

    while True:
        question = input("Ask a question (type 'exit' to quit): ")
        if question.lower() == 'exit':
            print("Exiting the program.")
            break
        
        # Ask the question and get the answer
        answer = ask_question(question, combined_text, api_key)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
