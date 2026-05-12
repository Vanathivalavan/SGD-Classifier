# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize model parameters (weights and bias) randomly.
2.Take one training sample at a time from the dataset.
3.Predict the output and calculate the error using a loss function.
4.Update the weights using stochastic gradient descent until the model converges.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: VANATHI.T
RegisterNumber:  212225040480
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SGDClassifier(max_iter=1000, tol=1e-3)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

## Output:
<img width="583" height="313" alt="WhatsApp Image 2026-05-11 at 2 34 18 PM" src="https://github.com/user-attachments/assets/34b4300b-0352-48ac-bb02-62dface230e8" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
