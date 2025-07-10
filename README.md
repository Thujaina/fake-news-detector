# 📰 Fake News Detector (Command-Line Version)

This project uses Machine Learning to detect whether a news article is **Fake** or **Real**. It's built using Python and runs in the terminal.

## 🛠 Technologies
- Python
- Pandas
- Scikit-learn
- TfidfVectorizer
- Logistic Regression
- Pickle

## 📦 Files
- `train_model.py` → Train the ML model
- `predict.py` → Predict using the saved model
- `Fake.csv`, `True.csv` → News datasets
- `fake_news_model.pkl` → Saved trained model
- `vectorizer.pkl` → Saved TF-IDF vectorizer

## 📈 How to Use

1. Place `Fake.csv` and `True.csv` in the project folder  
2. Run the training script:

3. Predict using:

4. Type/paste a news article when asked, and the prediction will be shown.

## ✅ Sample
Enter a news article text: The president met world leaders today to discuss climate change.
Prediction: 🟢 Real News

