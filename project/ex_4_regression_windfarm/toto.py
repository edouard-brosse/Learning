import numpy as np

# Chargement des données
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")  # Utilisation de ravel() pour convertir la structure
y_test = np.load("y_test.npy")

# Fonction pour entraîner le modèle de régression linéaire
def train_linear_regression(X, y):
    # Ajout d'une colonne de 1 pour le terme de biais
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    # Calcul des poids à l'aide de l'équation normale
    weights = np.linalg.inv(X.T @ X) @ X.T @ y
    return weights

# Fonction pour prédire les valeurs avec le modèle entraîné
def predict(X, weights):
    # Ajout d'une colonne de 1 pour le terme de biais
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    # Calcul des prédictions
    predictions = X @ weights
    return predictions

# Entraînement du modèle de régression linéaire
weights = train_linear_regression(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = predict(X_test, weights)

# Calcul du coefficient de détermination (R2 score)
def r2_score(y_true, y_pred):
    mean_y_true = np.mean(y_true)
    SS_res = np.sum((y_true - y_pred) ** 2)
    SS_tot = np.sum((y_true - mean_y_true) ** 2)
    r2 = 1 - (SS_res / SS_tot)
    return r2

# Calcul du R2 score
r2 = r2_score(y_test, y_pred)

print("R2 score :", r2)
