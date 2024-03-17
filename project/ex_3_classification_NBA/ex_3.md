Ce code effectue une recherche par grille pour trouver les meilleurs hyperparamètres d'un modèle de classification de forêt aléatoire. 
---
1. **Chargement des données** : Les données d'entraînement et de test sont chargées à partir de fichiers numpy.

2. **Initialisation du modèle** : Un modèle de classification de forêt aléatoire est initialisé. C'est un algorithme d'apprentissage automatique utilisé pour la classification.

3. **Définition de la grille des hyperparamètres** : Une grille des valeurs possibles des hyperparamètres du modèle est définie. Les hyperparamètres sont des paramètres du modèle qui ne sont pas appris directement à partir des données, mais qui affectent le processus d'apprentissage.

4. **Recherche par grille des meilleurs hyperparamètres** : Une recherche par grille est effectuée pour trouver la meilleure combinaison d'hyperparamètres. Cela implique l'entraînement et l'évaluation du modèle sur plusieurs combinaisons d'hyperparamètres, en utilisant une validation croisée pour évaluer les performances.

5. **Entraînement du modèle avec les meilleurs hyperparamètres** : Le modèle est ré-entraîné en utilisant les meilleurs hyperparamètres trouvés lors de la recherche par grille.

6. **Prédiction sur l'ensemble de test** : Le modèle entraîné est utilisé pour faire des prédictions sur l'ensemble de test.

7. **Calcul de la précision** : La précision des prédictions est calculée en comparant les étiquettes prédites avec les étiquettes réelles de l'ensemble de test.

8. **Affichage de la précision** : La précision est affichée sur le terminal.
