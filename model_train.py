import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

try:
    df = pd.read_csv('cleaned_phishing_emails.csv') #Cleaned dataset is loaded
except FileNotFoundError:
    print("Error: 'cleaned_phishing_emails.csv' not found.")
    print("Please run the data cleaning script first.")
    exit()

print("--- Successfully loaded cleaned dataset. ---")


# Define the features (email text) and the target (email type)
X = df['Email Text']
y = df['Email Type']         #Data preparation

# Split data into 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorize Text Data
# Convert text into numerical features using TF-IDF
print("--- Vectorizing text data... ---")
vectorizer = TfidfVectorizer(stop_words='english', max_features=7000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train the AI Model
print("--- Training the Logistic Regression model... ---")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
print("--- Model training complete. ---")

# 5. Evaluate the Model's Performance
print("\n--- Evaluating Model Performance ---")
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy on test data: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Save the Final Model and Vectorizer
print("\n--- Saving the final model and vectorizer... ---")
joblib.dump(model, 'phishing_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

print("✅ Model and vectorizer have been saved successfully!")