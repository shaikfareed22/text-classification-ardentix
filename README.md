# text-classification-ardentix

# 📄 README.md

## Text Classification System – AI/ML Engineer Intern Assignment

---

## 📌 Project Overview

This project is a **Text Classification System** developed as part of the **AI/ML Engineer Intern technical assignment**.

The system takes **raw text messages** as input and automatically classifies them into predefined categories.
In this project, the classification task is **Spam Detection**:

* **Spam** → Unwanted or promotional messages
* **Not Spam (Ham)** → Normal personal messages

The project demonstrates the **complete machine learning pipeline**, from data loading to model evaluation.

---

## 🎯 Objective

The main objective of this project is to:

* Understand how machines learn from text data
* Perform text preprocessing using NLP techniques
* Convert text into numerical features
* Train and evaluate machine learning models
* Build a working, end-to-end classification system

---

## 📂 Dataset Information

### Dataset Name

**SMS Spam Collection Dataset**

### Description

The dataset contains a collection of SMS messages, each labeled as:

* `spam`
* `ham` (not spam)

### Dataset Structure

| Column | Description        |
| ------ | ------------------ |
| v1     | Label (spam / ham) |
| v2     | SMS message text   |

### Example Data

| Label | Message              |
| ----- | -------------------- |
| spam  | Win a free prize now |
| ham   | Are we meeting today |

### Source

Publicly available dataset commonly used for text classification tasks.

---

## 🧹 Data Preprocessing Steps

Before training the model, the text data is cleaned and prepared:

1. Converted all text to lowercase
2. Removed numbers and special characters
3. Removed common stopwords (e.g., is, the, and)
4. Tokenized text into meaningful words

This ensures the model focuses only on **important information**.

---

## 🔢 Feature Extraction

### Method Used: **TF-IDF (Term Frequency – Inverse Document Frequency)**

TF-IDF converts text into numerical values by:

* Giving higher weight to important words
* Reducing the impact of very common words

This allows machine learning models to understand text mathematically.

---

## 🤖 Models Used

Two machine learning models were trained and compared:

### 1️⃣ Naive Bayes

* Fast and simple
* Commonly used for text classification

### 2️⃣ Logistic Regression (Final Model)

* Performs better on high-dimensional text data
* More stable and accurate than Naive Bayes

**Logistic Regression was chosen as the final model** due to higher accuracy.

---

## 📊 Model Evaluation

The models were evaluated using the following metrics:

* **Accuracy** – Overall correctness
* **Precision** – Correctness of spam predictions
* **Recall** – Ability to detect spam messages
* **F1-Score** – Balance between precision and recall

A **confusion matrix graph** was also generated to visually analyze performance.

---

## 🧪 Sample Prediction

Input:

```
"Congratulations! You won a free prize"
```

Output:

```
Spam
```

This shows the system works correctly on real-world text.

---

## 🛠 Application & Approach

### Application

* Spam detection for SMS messages
* Can be extended to:

  * Email spam detection
  * Sentiment analysis
  * Resume classification

### Approach

1. Load dataset
2. Clean and preprocess text
3. Convert text to numerical features (TF-IDF)
4. Train machine learning models
5. Evaluate and visualize results
6. Test with real input

---

## ⚙️ Setup Steps (GitHub Codespaces)

### 1️⃣ Clone Repository / Open Codespace

Open the repository using **GitHub Codespaces**.

### 2️⃣ Install Dependencies

Run the following command in terminal:

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Notebook

Open:

```
text_classifier.ipynb
```

Run all cells from top to bottom.

---

## 📁 Project Structure

```
text-classification-ardentix/
│
├── text_classifier.ipynb
├── spam.csv
├── requirements.txt
├── README.md
```

---

## 📌 Requirements

* Python 3.x
* pandas
* numpy
* scikit-learn
* nltk
* matplotlib
* seaborn

(All handled automatically in GitHub Codespaces)

---

## 🏁 Conclusion

This project demonstrates a complete **text classification system** using classical machine learning techniques.
It shows understanding of:

* Data preprocessing
* NLP fundamentals
* Model training and evaluation
* Practical implementation

---

## 🎯 Submission Note

This project was implemented using **GitHub Codespaces** to ensure a clean, reproducible, and error-free environment.
