import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
print("Dataset Preview:")
print(df.head())
sns.set_theme(style="whitegrid")
sns.pairplot(df.drop(columns=['species']), hue='species_name', palette='Set2')
plt.suptitle("Pairwise Relationships of Iris Features", y=1.02)
plt.show()
X = iris.data  
y = iris.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
print("Model training complete!")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"### Accuracy: {accuracy * 100:.2f}%\n")
print("### Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()