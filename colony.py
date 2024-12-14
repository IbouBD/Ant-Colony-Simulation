import numpy as np

# Define ant behavor
class Ant:
    
    def __init__(self, x, y, env):
        self.pos = np.array([float(x), float(y)])
        self.old_pos = np.array([None, None])
        self.update_vec_director = None
        self.has_food = False
        self.is_dead = False
        self.fitness = 0  # Fitness pour NEAT
        self.type = None,
        self.deposit_pheromone = False
        self.pheromone_type = None
        self.surrounding_pheromones = None
        self.inputs = []
        self.radius = 5 # rayon de perception des pheromones
        self.pickup_location = tuple
        self.idle_time = 0 # Temps d'inactivité

    def update_idle_time(self): 
        if not self.has_food and not self.is_dead: 
            self.idle_time += 1 
        else: self.idle_time = 0

    def move(self, dx, dy):

        self.pos += np.array([float(dx), float(dy)])

    def get_inputs(self, env):
        """Retourne les entrées pour le réseau NEAT en utilisant la grille NumPy."""
        x, y = int(self.pos[0]), int(self.pos[1])
        
        # Limites de la zone de perception dans la grille
        x_min = max(0, x - self.radius)
        x_max = min(env.pheromone_system.width, x + self.radius + 1)
        y_min = max(0, y - self.radius)
        y_max = min(env.pheromone_system.height, y + self.radius + 1)

        # Extraire la sous-grille de phéromones
        local_pheromones = env.pheromone_system.pheromones[y_min:y_max, x_min:x_max, :]

        # Somme des phéromones par type dans la zone de perception
        surrounding_pheromones = np.sum(local_pheromones, axis=(0, 1))

        # Construire le vecteur d'entrées pour le réseau
        self.inputs = [
            self.pos[0],  # Position x actuelle
            self.pos[1],  # Position y actuelle
            1 if self.has_food else 0,  # Si la fourmi transporte de la nourriture
        ] + surrounding_pheromones.tolist()  # Ajouter les niveaux de phéromones par type

        return self.inputs


    
    def state(self):
        return self.inputs # size = 6