import fitz
import pickle
import re
import google.generativeai as genai
import numpy as np

# Configure Gemini API (Use Gemini 1.5 Flash for free-tier access)
GEMINI_API_KEY = "AIzaSyB490mibuYnyw4Fe4Z62ZW3Oc2_Af_d6wM"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

# Function to extract text from PDF
def read_pdf_pymupdf(file_path):
    document = fitz.open(file_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text.strip()

# Function to remove stopwords
def remove_stopwords(text):
    stop_words = set([
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
        "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
        "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
        "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
        "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
        "at", "by", "for", "with", "about", "against", "between", "into", "through",
        "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "any",
        "both", "each", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
        "t", "can", "will", "just", "don", "should", "now"
    ])
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Function to clean the resume text
def clean_resume(txt):
    clean_text = re.sub('http\S+\s', ' ', txt)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+\s', ' ', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return remove_stopwords(clean_text)

# Load trained classifier and TF-IDF vectorizer
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Define category mapping
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and Fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

# Function to predict using Gemini 1.5 Flash (if ML model is unsure)
def predict_with_gemini(text):
    prompt = f"""
    You are an AI resume classification expert. The following is a resume:

    ---
    {text}
    ---

    Based on the content, classify this resume into one of the following categories:
    - Java Developer
    - Testing
    - DevOps Engineer
    - Python Developer
    - Web Designing
    - HR
    - Hadoop
    - Blockchain
    - ETL Developer
    - Operations Manager
    - Data Science
    - Sales
    - Mechanical Engineer
    - Arts
    - Database
    - Electrical Engineering
    - Health and Fitness
    - PMO
    - Business Analyst
    - DotNet Developer
    - Automation Testing
    - Network Security Engineer
    - SAP Developer
    - Civil Engineer
    - Advocate

    **Provide only the category name as output** (no additional explanation).
    """

    # Use Gemini 1.5 Flash for better classification if needed
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    return response.text.strip()

# Function to predict category (ML + Gemini)
def predict_resume_category(file_path, confidence_threshold=0.8):
    try:
        # Read and clean the resume text
        text = read_pdf_pymupdf(file_path)
        cleaned_text = clean_resume(text)

        # Transform the cleaned resume using the trained TfidfVectorizer
        input_features = tfidf.transform([cleaned_text])

        # Predict using ML classifier
        prediction_probabilities = clf.predict_proba(input_features)
        confidence_score = np.max(prediction_probabilities)
        prediction_id = np.argmax(prediction_probabilities)

        # Get the category name from mapping
        predicted_category = category_mapping.get(prediction_id, "Unknown")

        # Print the confidence score
        print(f"ML Model Prediction: {predicted_category} (Confidence: {confidence_score:.2f})")

        # If confidence is lower than the threshold, use Gemini AI for classification
        if confidence_score < confidence_threshold:
            print("🔍 Confidence is low (<0.8), using Gemini AI for better classification...")
            return predict_with_gemini(cleaned_text)

        return predicted_category
    
    except Exception as e:
        return f"Error: {str(e)}"

# **Take PDF path as input from the user**
file_path = input("Enter the full path of the resume PDF file: ")

# Predict category
predicted_category = predict_resume_category(file_path)
print(f"Final Predicted Resume Category: {predicted_category}")
