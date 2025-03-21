import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error, roc_auc_score

# Load the dataset
df = pd.read_csv("UpdatedResumeDataSet.csv")

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

# Function to clean resumes
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    cleanText = remove_stopwords(cleanText)
    return cleanText

# Clean the resumes
df['Resume'] = df['Resume'].apply(lambda x: cleanResume(x))

# Label encoding for the categories
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
resumeTextTFIDF = tfidf.fit_transform(df['Resume'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(resumeTextTFIDF, df['Category'], test_size=0.2, random_state=42)

# Train the model using Random Forest
clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(X_train, y_train)

# Predict the test set results
ypred_rf = clf_rf.predict(X_test)

# Calculate the metrics
accuracy_rf = accuracy_score(y_test, ypred_rf)
precision_rf = precision_score(y_test, ypred_rf, average='weighted')
recall_rf = recall_score(y_test, ypred_rf, average='weighted')
f1_rf = f1_score(y_test, ypred_rf, average='weighted')
mse_rf = mean_squared_error(y_test, ypred_rf)
mae_rf = mean_absolute_error(y_test, ypred_rf)
roc_auc_rf = roc_auc_score(y_test, clf_rf.predict_proba(X_test), multi_class='ovr')

# Print the metrics
print(f"Random Forest Classifier Metrics:")
print(f"Accuracy: {accuracy_rf}")
print(f"Precision: {precision_rf}")
print(f"Recall: {recall_rf}")
print(f"F1-Score: {f1_rf}")


# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, ypred_rf)
plt.figure(figsize=(5,5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Feature Importance
feature_importances = clf_rf.feature_importances_
feature_names = tfidf.get_feature_names_out()
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display the top 20 important features
top_features = feature_importance_df.head(20)
print(top_features)

# Plot the top 20 features
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_features)
plt.title('Top 20 Important Features')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Save the model and vectorizer
pickle.dump(tfidf, open('tfidf1.pkl', 'wb'))
pickle.dump(clf_rf, open('clf_rf1.pkl', 'wb'))
