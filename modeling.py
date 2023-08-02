import pandas as pd
import numpy as np
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix

iris = pd.read_csv("./Data/iris.csv")

X = iris[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = iris["Species"]

# Train and fit model
knn = KNeighborsClassifier(n_neighbors=5).fit(X, y)


# make predictions
y_predict = knn.predict(X)

# Check results
print(confusion_matrix(y, y_predict))
print(classification_report(y, y_predict))

# Save the model

# Saving model to disk
pickle.dump(knn, open("model.pkl", 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))

print(model.predict([[5.0, 3.0, 1.5, 0.2]]))
