# 🎣 Phishing Email Detector

A machine learning-powered web application that detects phishing emails using Natural Language Processing and Logistic Regression. Built with Python, Streamlit, and scikit-learn.

## 🚀 Features

- **Real-time Detection**: Instantly classify emails as Safe or Phishing
- **High Accuracy**: Trained on 175,000+ email samples
- **Web Interface**: User-friendly Streamlit application
- **Confidence Scoring**: Shows prediction confidence percentages
- **Modular Design**: Separate scripts for data processing, training, and prediction

## 📊 Project Overview

This project implements a complete machine learning pipeline for phishing email detection:

1. **Data Processing**: Clean and preprocess email dataset
2. **Feature Engineering**: TF-IDF vectorization for text analysis
3. **Model Training**: Logistic Regression classifier
4. **Web Deployment**: Interactive Streamlit application
5. **Standalone Prediction**: Command-line prediction script

## 🛠️ Technology Stack

- **Python3**
- **Machine Learning**: scikit-learn, pandas, numpy
- **Web Framework**: Streamlit
- **Text Processing**: TF-IDF Vectorization
- **Model Persistence**: joblib

## 📁 Project Structure

```
Phishing Detector/
├── app.py                      # Streamlit web application
├── data_checker.py             # Data exploration script
├── data_cleaning.py            # Data preprocessing
├── model_train.py              # Model training script
├── predict.py                  # Standalone prediction
├── Phishing_Email.csv          # Original dataset
├── cleaned_phishing_emails.csv # Processed dataset
├── AI_Models/
│   ├── phishing_model.joblib   # Trained model
│   └── vectorizer.joblib       # TF-IDF vectorizer
└── README.md                   # Project documentation
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install streamlit pandas scikit-learn joblib numpy
```

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd "Phishing Detector"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Option 1: Web Application (Recommended)

1. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Paste email content** and click "Check Email"

#### Option 2: Command Line Prediction

```bash
python predict.py
```

## 🔧 Model Training (Optional)

If you want to retrain the model:

1. **Check the dataset**
   ```bash
   python data_checker.py
   ```

2. **Clean the data**
   ```bash
   python data_cleaning.py
   ```

3. **Train the model**
   ```bash
   python model_train.py
   ```

## 📈 Model Performance

- **Algorithm**: Logistic Regression
- **Features**: 7,000 TF-IDF features
- **Training Data**: ~140,000 emails (80% split)
- **Test Data**: ~35,000 emails (20% split)
- **Evaluation**: Accuracy, Precision, Recall, F1-Score

## ☕ Support the Project

If this project helped you or you found it useful, consider buying me a coffee! Your support helps maintain and improve this project.

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-support%20my%20work-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/sinanfiroz)

**Other ways to support:**
- ⭐ Star this repository
- 🍴 Fork and contribute 
- 📢 Share with others
- 🐛 Report issues and bugs


## 🎯 How It Works

### 1. Text Preprocessing
- Removes stop words (common English words)
- Converts text to lowercase
- Handles missing data

### 2. Feature Extraction
- **TF-IDF Vectorization**: Converts email text to numerical features
- **Term Frequency**: How often words appear in each email
- **Inverse Document Frequency**: How rare words are across all emails

### 3. Classification
- **Logistic Regression**: Binary classifier (Safe vs Phishing)
- **Probability Scores**: Confidence percentages for predictions

### 4. Prediction Pipeline
```
Email Text → TF-IDF Transform → Model Prediction → Result + Confidence
```

## 📝 Example Usage

### Web Interface
1. Paste suspicious email content
2. Click "Check Email"
3. Get instant result with confidence score

### Sample Inputs

**Phishing Email Example:**
```
Congratulations! You've won $1000! Click here immediately to claim your prize!
```

**Safe Email Example:**
```
Hi John, just confirming our meeting tomorrow at 2 PM. Please let me know if this works.
```

## 🔍 Key Features Explained

### TF-IDF Vectorization
- **Purpose**: Convert text to numbers for machine learning
- **Max Features**: 7,000 most important words
- **Stop Words**: Removes common words like "the", "and", "is"

### Logistic Regression
- **Why**: Fast, interpretable, good for text classification
- **Output**: Binary classification with probability scores
- **Training**: 1000 max iterations for convergence

### Model Persistence
- **Saved Files**: Both model and vectorizer saved separately
- **Consistency**: Ensures same preprocessing for new predictions

## 🚨 Security Considerations

- Input validation for email content
- No storage of user email data
- Secure model file handling
- Error handling for corrupted files

## 🔧 Suggested Customizations

### Modify TF-IDF Parameters
```python
# In model_train.py
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=7000,  # Adjust number of features
    ngram_range=(1, 2)  # Add bigrams
)
```

### Change Model Algorithm
```python
# Replace LogisticRegression with other algorithms
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
```

## 📊 Performance Monitoring

The model provides:
- **Prediction confidence** (probability scores)
- **Classification reports** during training
- **Accuracy metrics** on test data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🆘 Troubleshooting

### Common Issues

1. **"Model files not found" error**
   - Run `python model_train.py` first to create model files

2. **Import errors**
   - Install missing packages: `pip install package_name`

3. **Dataset not found**
   - Ensure `Phishing_Email.csv` is in the project directory

4. **Streamlit not opening**
   - Check if port 8501 is available
   - Try: `streamlit run app.py --server.port 8502`

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**⚠️ Disclaimer**: This tool is for educational and research purposes. Always use multiple layers of security for email protection in production environments.
