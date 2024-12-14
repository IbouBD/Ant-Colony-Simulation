
# Simulateur de Comportement Collectif des Fourmis avec NEAT

Ce projet simule le comportement collectif d'une colonie de fourmis dans un environnement dynamique à l'aide de l'algorithme NEAT (NeuroEvolution of Augmenting Topologies). Les fourmis évoluent pour apprendre à résoudre une tâche clé : ramasser tous les fruits dans leur environnement et les ramener à la colonie, sans connaissance préalable de la disposition des fruits ni du nombre total.

## Structure du Projet

- `README.md` : Documentation et explication du projet.
- `colony.py` : Contient la logique de la colonie de fourmis et leur comportement.
- `config.txt` : Fichier de configuration pour l'algorithme NEAT.
- `simulation.py` : Code principal pour la simulation et l'évolution des fourmis.

## Prérequis

- Python 3.x
- Bibliothèques nécessaires :
  - `neat-python`
  - `numpy`
  - `pygame`
  - `pickle`

Pour installer les bibliothèques nécessaires, utilisez la commande suivante :

```bash
pip install neat-python numpy pygame pickle
```

## Utilisation

1. **Configuration** : Assurez-vous que le fichier `config.txt` est configuré correctement pour vos besoins.

2. **Lancer la Simulation** :

   ```bash
   python simulation.py
   ```
## Fonctionnalités

- **Environnement Dynamique** : Les fourmis évoluent dans un environnement où la disposition des fruits est inconnue.
- **Évolution Neuro-Evolution** : Utilisation de l'algorithme NEAT pour faire évoluer les comportements des fourmis.
- **Fitness Adaptatif** : Fonction de fitness pour encourager la collecte de fruits, la collaboration et l'exploration efficace.

## Auteurs

- [IbouBD] - Créateur du projet.
