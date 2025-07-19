import joblib

# Loading the saved model and vectorizer
try:
    model = joblib.load('AI_Models/phishing_model.joblib')
    vectorizer = joblib.load('AI_Models/vectorizer.joblib')
except FileNotFoundError:
    print("Error: Model files not found.")
    print("Please run the 'train_model.py' script first.")
    exit()

def predict_email_type(email_text):
  """
  Classifies an email text as 'Phishing Email' or 'Safe Email'.
  """
  # Transform the input text using the loaded vectorizer
  email_tfidf = vectorizer.transform([email_text])
  
  # Predict using the loaded model
  prediction = model.predict(email_tfidf)
  
  # Return the prediction
  return prediction[0]

# Example 1: A suspicious email
suspicious_email = "congratulations you won a 1000 dollar walmart gift card go to http://tinyurl.com/scamlink to claim now"

# Example 2: A normal email
safe_email = "Hi Bob, just wanted to confirm our meeting for tomorrow at 2 PM. Please let me know if the time still works for you. Thanks, Alice."

print("--- Classifying the Emails ---")

prediction1 = predict_email_type(suspicious_email)
print(f"\nEmail Text: '{suspicious_email}'")
print(f"Predicted Type: {prediction1} 🤖")

prediction2 = predict_email_type(safe_email)
print(f"\nEmail Text: '{safe_email}'")
print(f"Predicted Type: {prediction2} ✅")