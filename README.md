# Customer Review Sentiment Classifier

An end-to-end Machine Learning web application designed to automatically analyze customer review text and classify sentiment as **Positive** or **Negative** using Natural Language Processing (NLP) techniques and Supervised Learning algorithms.

---

## Aim & Objectives

* **Aim:** To build a web-based sentiment analysis tool that classifies unstructured customer review text in real-time.
* **Objectives:**
  1. Preprocess raw textual data (tokenization, stop-word removal, TF-IDF vectorization).
  2. Train and evaluate a probabilistic machine learning model (Multinomial Naive Bayes / Logistic Regression).
  3. Develop an intuitive, modern, web interface using Flask, HTML5, CSS3, and JavaScript.
  4. Display real-time sentiment predictions alongside confidence scores and performance analytics.

---

## Tech Stack & Algorithms

* **Programming Language:** Python 3.8+
* **Machine Learning & NLP:** `scikit-learn`, `pandas`, `numpy`
* **Backend Framework:** Flask
* **Frontend Technologies:** HTML5, CSS3, JavaScript (ES6+), Chart.js
* **Algorithms Used:**
  * **TF-IDF Vectorizer:** Feature extraction technique converting raw text into numerical representation based on term frequency and inverse document frequency.
  * **Multinomial Naive Bayes:** Probabilistic classifier suitable for high-dimensional text classification tasks.

---

## Project Directory Structure

```text
Customer_Review_Classifier/
│
├── app.py                   # Flask backend & model pipeline execution
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
└── templates/
    └── index.html           # Web application frontend UI
