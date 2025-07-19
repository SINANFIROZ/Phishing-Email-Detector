import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# Load the dataset
try:
    df = pd.read_csv('Phishing_Email.csv')
except FileNotFoundError:
    print("'Phishing_Email.csv' not found. Please make sure the file is in the same directory.")
    exit()

# Clean the data
# Dropping the unnecessary column
df = df.drop(columns=['Unnamed: 0'])
# Drop rows with missing email text
df = df.dropna(subset=['Email Text'])

# Check the cleaned data info
print("CLEANED DATASET INFO:")
print(df.info())


# Prepare data for the model
# 'X' is the feature (the email text)
# 'y' is the label (the email type)
X = df['Email Text']
y = df['Email Type']

# Split data into training and testing sets
# We train the model on 80% of the data and test its performance on the remaining 20%.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
# This converts the text into a matrix of TF-IDF features.
# 'stop_words='english'' removes common English words like 'the', 'a', 'is'.
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Fit the vectorizer on the training data and transform it
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
X_test_tfidf = vectorizer.transform(X_test)

print("\n--- Data Preparation Complete ---")
print(f"Training data shape: {X_train_tfidf.shape}")
print(f"Test data shape: {X_test_tfidf.shape}")

df.to_csv('cleaned_phishing_emails.csv', index=False)

print("\nCleaned data has been saved to 'cleaned_phishing_emails.csv'")