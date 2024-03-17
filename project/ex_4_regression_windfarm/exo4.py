import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import joblib

# Charger les données d'entraînement
X_train = np.load('X_train.npy')
y_train = np.load('Y_train.npy').ravel()

# Charger les données de test
X_test = np.load('X_test.npy')
y_test = np.load('Y_test.npy').ravel()

# Standardisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialiser et entraîner le modèle Ridge
ridge_model = Ridge(alpha=0.2)  # Paramètre alpha à ajuster
ridge_model.fit(X_train_scaled, y_train)

# Initialiser et entraîner le modèle MLP
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)  # Paramètres à ajuster
mlp_model.fit(X_train_scaled, y_train)

# Faire des prédictions sur l'ensemble de test
ridge_pred = ridge_model.predict(X_test_scaled)
mlp_pred = mlp_model.predict(X_test_scaled)

# Calculer le score R2
ridge_r2 = r2_score(y_test, ridge_pred)
mlp_r2 = r2_score(y_test, mlp_pred)

print("Ridge R2 Score:", ridge_r2)
print("MLP R2 Score:", mlp_r2)
