import streamlit as st
import joblib

# Load the Saved Model and Vectorizer
try:
    model = joblib.load('AI_Models/phishing_model.joblib')
    vectorizer = joblib.load('AI_Models/vectorizer.joblib')
except FileNotFoundError:
    st.error("Model files not found. Please run the training script first.")
    st.stop()

# Web App Interface

# Set the title and a small description
st.title("Phishing Email Detector 🎣")
st.write("Paste the content of an email below to check if it's a potential phishing attempt.")

# Create a text area for user input
email_text = st.text_area("Email Content", height=200)

# Create a button to trigger the prediction
if st.button("Check Email"):
    if email_text:
        # Transform the input text using the loaded vectorizer
        text_tfidf = vectorizer.transform([email_text])
        
        # Predict using the loaded model
        prediction = model.predict(text_tfidf)
        prediction_proba = model.predict_proba(text_tfidf)

        # Display the result
        st.subheader("Result:")
        if prediction[0] == "Phishing Email":
            st.warning("This looks like a Phishing Email! 🚨")
            st.write(f"Confidence: {prediction_proba[0][0]:.2%}")
        else:
            st.success("This appears to be a Safe Email. ✅")
            st.write(f"Confidence: {prediction_proba[0][1]:.2%}")
    else:
        st.info("Please paste some email content to check.")