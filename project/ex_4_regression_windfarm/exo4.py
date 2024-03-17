import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
import joblib

# Charger les données d'entraînement
X_train = np.load('X_train.npy')
y_train = np.load('Y_train.npy').ravel()  # Utiliser ravel() pour convertir y en un tableau 1D

# Charger les données de test
X_test = np.load('X_test.npy')
y_test = np.load('Y_test.npy').ravel()  # Utiliser ravel() pour convertir y en un tableau 1D

# Charger le StandardScaler et ajuster aux données d'entraînement
scaler = StandardScaler()
scaler.fit(X_train)

# Transformer les données d'entraînement et de test
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialiser et entraîner les modèles
ridge_model = Ridge(alpha=0.1)
lasso_model = Lasso(alpha=0.1)
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
svr_model = SVR()
ada_boost_model = AdaBoostRegressor()

# Entraîner les modèles
ridge_model.fit(X_train_scaled, y_train)
lasso_model.fit(X_train_scaled, y_train)
mlp_model.fit(X_train_scaled, y_train)
svr_model.fit(X_train_scaled, y_train)
ada_boost_model.fit(X_train_scaled, y_train)

# Faire des prédictions sur l'ensemble de test
ridge_pred = ridge_model.predict(X_test_scaled)
lasso_pred = lasso_model.predict(X_test_scaled)
mlp_pred = mlp_model.predict(X_test_scaled)
svr_pred = svr_model.predict(X_test_scaled)
ada_boost_pred = ada_boost_model.predict(X_test_scaled)

# Calculer le score R2 pour chaque modèle
ridge_r2 = r2_score(y_test, ridge_pred)
lasso_r2 = r2_score(y_test, lasso_pred)
mlp_r2 = r2_score(y_test, mlp_pred)
svr_r2 = r2_score(y_test, svr_pred)
ada_boost_r2 = r2_score(y_test, ada_boost_pred)

# Afficher les scores R2
print("Ridge R2 Score:", ridge_r2)
print("Lasso R2 Score:", lasso_r2)
print("MLP R2 Score:", mlp_r2)
print("SVR R2 Score:", svr_r2)
print("AdaBoost R2 Score:", ada_boost_r2)
