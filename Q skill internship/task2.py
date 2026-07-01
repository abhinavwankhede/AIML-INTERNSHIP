import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
df = pd.read_csv(url, sep='\t', names=['label', 'message'], compression='zip')
print("Dataset Preview:")
print(df.head())
print(f"\nDataset Shape: {df.shape}")
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label_num']
print(f"Vocabulary size (unique words): {X.shape[1]}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
spam_model = MultinomialNB()
spam_model.fit(X_train, y_train)
print("Spam detector model successfully trained!")
y_pred = spam_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"### Performance Metrics:")
print(f"- **Accuracy**:  {accuracy * 100:.2f}%")
print(f"- **Precision**: {precision * 100:.2f}%  <-- (How many predicted spam emails were actually spam)")
print(f"- **Recall**:    {recall * 100:.2f}%  <-- (How many total spam emails were successfully caught)")
print(f"- **F1-Score**:  {f1 * 100:.2f}%\n")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Spam Detector Confusion Matrix')
plt.show()
def predict_message(text):
    text_vectorized = vectorizer.transform([text])
    prediction = spam_model.predict(text_vectorized)
    return "SPAM 🚨" if prediction[0] == 1 else "HAM (Safe) ✅"
print(predict_message("Hey, are we still meeting up for lunch today at 1 PM?"))
print(predict_message("CONGRATULATIONS! You have won a free $1000 Walmart gift card. Call now to claim your prize!"))