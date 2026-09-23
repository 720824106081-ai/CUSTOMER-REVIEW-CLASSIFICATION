from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Sample dataset for model training
corpus = [
    ("Great product! Absolutely loved the quality and packaging.", "Positive"),
    ("Terrible experience. Item arrived broken and late.", "Negative"),
    ("Fast shipping, excellent customer support, works perfectly.", "Positive"),
    ("Waste of money. Quality is cheap and broke on day one.", "Negative"),
    ("Highly recommended! Fits great and high quality material.", "Positive"),
    ("Worst purchase I have ever made. Extremely disappointed.", "Negative"),
    ("Decent value for money, satisfied with the purchase.", "Positive"),
    ("Poor customer service and bad product quality.", "Negative")
]

# Separate features and labels
X_train_text, y_train = zip(*corpus)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train_text)

# Train Multinomial Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    review_text = ""
    
    if request.method == "POST":
        review_text = request.form.get("review", "")
        if review_text.strip():
            # Vectorize user input and predict
            text_vec = vectorizer.transform([review_text])
            prediction = model.predict(text_vec)[0]

    return render_template("index.html", prediction=prediction, review=review_text)

if __name__ == "__main__":
    app.run(debug=True)