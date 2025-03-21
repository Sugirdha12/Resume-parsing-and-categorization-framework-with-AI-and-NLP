from PyPDF2 import PdfReader  
import requests
import os
import sys

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def ask_question(question, context, api_key):
    url = "https://api.cohere.ai/chat"  # Update to chat endpoint
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": f"Resume Content:\n{context}\n\nQuestion: {question}",
        "temperature": 0.1,
        "max_tokens": 500
    }
    response = requests.post(url, headers=headers, json=payload)
    response_json = response.json()

    try:
        answer = response_json.get('text', "Sorry, I couldn't find the answer.").strip()
    except KeyError:
        answer = "Sorry, I couldn't find the answer."

    return answer



def main():
    pdf_path = input("Enter the path to the PDF file: ")
    if not os.path.exists(pdf_path):
        print("The specified file does not exist. Please check the path and try again.")
        return
    if not pdf_path.lower().endswith('.pdf'):
        print("Invalid file format. Please enter the path of a PDF file.")
        return

    # Extract text from the PDF file
    resume_text = extract_text_from_pdf(pdf_path)

    # Your Cohere API key
    api_key = 'KfRekS4l6ZtIsjIhiu8SMjAVnz3OXsMz5X5S6GP2'

    while True:
        question = input("Ask a question (type 'exit' to quit): ")
        if question.lower() == 'exit':
            print("Exiting the program.")
            break  # Break out of the loop to exit gracefully
        
        answer = ask_question(question, resume_text, api_key)
        print(answer)

    # Ensure the program exits completely
    sys.exit()  # Explicitly terminate the program

if __name__ == "__main__":
    main()
