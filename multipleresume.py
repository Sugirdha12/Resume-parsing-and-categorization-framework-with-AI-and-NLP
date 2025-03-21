from PyPDF2 import PdfReader
import requests
import os

# Function to extract text from a single PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

# Function to ask a question using Cohere Chat API and get a response
def ask_question(question, context, api_key):
    url = "https://api.cohere.ai/chat"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": f"Resume Content:\n{context}\n\nQuestion: {question}",
        "temperature": 0.5,
        "max_tokens": 500
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        response_json = response.json()
        answer = response_json.get("text", "Sorry, I couldn't find the answer.").strip()
        return answer
    except requests.exceptions.RequestException as e:
        print(f"Error with Cohere API request: {e}")
        return "Sorry, there was an issue with the API request."

# Function to extract text from all PDFs in a directory
def extract_texts_from_pdfs(directory):
    pdf_texts = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            pdf_texts[filename] = extract_text_from_pdf(pdf_path)
    return pdf_texts

# Main function
def main():
    # Ask user for the directory containing PDFs
    pdf_directory = input("Enter the path to the directory containing PDF files: ")
    
    # Validate the directory exists
    if not os.path.isdir(pdf_directory):
        print(f"The specified directory does not exist: {pdf_directory}")
        return
    
    # Hardcoded API key
    api_key = 'KfRekS4l6ZtIsjIhiu8SMjAVnz3OXsMz5X5S6GP2'
    
    # Extract text from all PDFs in the given directory
    pdf_texts = extract_texts_from_pdfs(pdf_directory)

    while True:
        # Display available PDFs
        print("\nAvailable PDFs:")
        for i, filename in enumerate(pdf_texts.keys(), 1):
            print(f"{i}. {filename}")
        
        # Ask user to select a PDF
        selected_index = input("Select a PDF by number (type 'exit' to quit): ")
        if selected_index.lower() == 'exit':
            break
        
        try:
            selected_index = int(selected_index) - 1
            selected_filename = list(pdf_texts.keys())[selected_index]
            resume_text = pdf_texts[selected_filename]
        except (ValueError, IndexError):
            print("Invalid selection. Please try again.")
            continue

        while True:
            # Ask user to ask a question
            question = input(f"Ask a question about {selected_filename} (type 'exit' to quit): ")
            if question.lower() == 'exit':
                break

            # Get answer from Cohere Chat API
            answer = ask_question(question, resume_text, api_key)
            print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
