import pickle

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict(news):
    vec = vectorizer.transform([news])
    pred = model.predict(vec)
    return "🟢 Real News" if pred[0] == 1 else "🔴 Fake News"

if __name__ == "__main__":
    print("📰 Fake News Detector")
    print("---------------------")
    news_input = input("Enter a news article text: ")
    result = predict(news_input)
    print(f"\nPrediction: {result}")
