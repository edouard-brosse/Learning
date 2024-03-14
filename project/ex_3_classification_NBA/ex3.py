import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Initialiser le modèle de régression logistique
model = LogisticRegression()

# Entraîner le modèle
model.fit(X_train, y_train)

# Prédire sur l'ensemble de test
predictions = model.predict(X_test)

# Calculer la précision
exactitude = accuracy_score(y_test, predictions)
print("Précision :", exactitude)
