from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Charger le jeu de données Iris
iris = load_iris()
X, y = iris.data, iris.target

# Entraîner le modèle
model = RandomForestClassifier()
model.fit(X, y)

# Sauvegarder le modèle entraîné
with open('iris_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
