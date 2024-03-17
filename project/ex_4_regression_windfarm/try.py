from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import numpy as np

# Chargement des données
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy").ravel()  # Utilisation de ravel() pour convertir la structure
y_test = np.load("y_test.npy").ravel()    # Utilisation de ravel() pour convertir la structure

# Initialisation et entraînement du modèle Ridge
ridge_model = Ridge(alpha=1.0)  # Vous pouvez ajuster l'hyperparamètre alpha si nécessaire
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
r2_ridge = r2_score(y_test, y_pred_ridge)
print("R2 score pour Régression Ridge :", r2_ridge)

# Initialisation et entraînement du modèle SVR
svr_model = SVR(kernel='linear')  # Vous pouvez essayer différents noyaux (linear, rbf, etc.)
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)
r2_svr = r2_score(y_test, y_pred_svr)
print("R2 score pour Support Vector Regressor :", r2_svr)
