import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Charger les données
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Initialiser le modèle de forêt aléatoire
model = RandomForestClassifier()

# Définir la grille des hyperparamètres
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Recherche par grille des meilleurs hyperparamètres
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Meilleurs hyperparamètres trouvés
best_params = grid_search.best_params_

# Entraîner le modèle avec les meilleurs hyperparamètres
best_model = RandomForestClassifier(**best_params)
best_model.fit(X_train, y_train)

# Prédire sur l'ensemble de test
predictions = best_model.predict(X_test)

# Calculer la précision
exactitude = accuracy_score(y_test, predictions)
print("Précision :", exactitude)

