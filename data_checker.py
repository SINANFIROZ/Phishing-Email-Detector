import pandas as pd
# Load the dataset
try:
    df = pd.read_csv('Phishing_Email.csv')
except FileNotFoundError:
    print("'Phishing_Email.csv' not found. Please make sure the file is in the same directory.")
    exit()

# Let's see what our data looks like
print("--- Dataset Head ---")
print(df.head())

# Check for missing values and get info
print("\n--- Dataset Info ---")
print(df.info())

# See the distribution of 'Phishing' vs 'Safe' emails
print("\n--- Email Type Distribution ---")
print(df['Email Type'].value_counts())