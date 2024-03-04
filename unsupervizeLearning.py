from sklearn.cluster import KMeans
import numpy as np

# Données d'entrée
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

# Création de l'objet KMeans avec 2 clusters
kmeans = KMeans(n_clusters=2)

# Entraînement de l'algorithme
kmeans.fit(X)

# Prédiction des clusters pour les données d'entrée
labels = kmeans.predict(X)

# Affichage des résultats
print(labels)