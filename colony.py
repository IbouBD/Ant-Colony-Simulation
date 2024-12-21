import numpy as np
from icecream import ic

# Define ant behavor
class Ant:
    
    def __init__(self, x, y, env):
        self.pos = np.array([float(x), float(y)])
        self.old_pos = np.array([None, None])
        self.update_vec_director = None
        self.visited_positions = set()  # Positions explorées
        self.has_food = False
        self.is_dead = False
        self.fitness = 0  # Fitness pour NEAT
        self.type = None,
        self.deposit_pheromone = False
        self.pheromone_type = None
        self.surrounding_pheromones = None
        self.inputs = []
        self.radius = 5 # rayon de perception des pheromones
        self.search_radius = 15  # Rayon de recherche limité
        self.pickup_location = tuple
        self.idle_time = 0 # Temps d'inactivité
        self.env = env

    def update_idle_time(self): 
        if not self.has_food and not self.is_dead: 
            self.idle_time += 1 
        else: self.idle_time = 0

    def move(self, dx, dy):

        self.pos += np.array([float(dx), float(dy)])

    def get_closest_food_direction(self, env):
        """
        Trouve la direction vers la nourriture la plus proche.
        
        Returns:
            list: [dx, dy] vecteur normalisé pointant vers la nourriture la plus proche,
                [0, 0] si aucune nourriture n'est trouvée.
        """
        x, y = int(self.pos[0]), int(self.pos[1])
        min_distance = float('inf')
        closest_food = None
        
        
        # Limites de la zone de recherche
        x_min = max(0, x - self.search_radius)
        x_max = min(env.width, x + self.search_radius)
        y_min = max(0, y - self.search_radius)
        y_max = min(env.height, y + self.search_radius)
        
        # Recherche de nourriture dans la zone délimitée
        food_positions = np.where(env.food[y_min:y_max, x_min:x_max] == 2)
        
        if len(food_positions[0]) > 0:
            # Convertir en coordonnées relatives
            food_y = food_positions[0] + y_min
            food_x = food_positions[1] + x_min
            
            # Calculer les distances pour chaque position de nourriture
            for fx, fy in zip(food_x, food_y):
                distance = np.sqrt((fx - x)**2 + (fy - y)**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_food = (fx, fy)
            
            if closest_food:
                # Calculer le vecteur direction
                dx = closest_food[0] - x
                dy = closest_food[1] - y
                
                # Normaliser le vecteur
                magnitude = np.sqrt(dx**2 + dy**2)
                if magnitude > 0:
                    dx = dx / magnitude
                    dy = dy / magnitude
                    
                return [dx, dy]
        
        return [0, 0]  # Aucune nourriture trouvée

    def get_inputs(self, env, ants):
        """Retourne les entrées pour le réseau NEAT en utilisant la grille NumPy, optimisée."""
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

        # Calculer la distance à la colonie (une seule fois)
        colony_center = np.array([env.width / 2, env.height / 2])
        distance_to_colony = np.linalg.norm(self.pos - colony_center)

        # Obtenir les positions des autres fourmis
        ant_positions = np.array([ant.pos for ant in ants if ant != self])  # Exclure la fourmi actuelle
        if len(ant_positions) > 0:
            distances = np.linalg.norm(ant_positions - self.pos, axis=1)  # Calcul vectorisé des distances
            closest_indices = np.argsort(distances)[:3]  # Indices des 3 plus proches voisins
            closest_ants = ant_positions[closest_indices]
        else:
            # Si aucune autre fourmi, remplir avec des zéros
            closest_ants = np.zeros((3, 2))

        # Ajouter information sur la nourriture proche
        food_direction = self.get_closest_food_direction(env)

        # Construire le vecteur d'entrées pour le réseau
        self.inputs = [
            self.pos[0],  # Position x actuelle
            self.pos[1],  # Position y actuelle
            *closest_ants.flatten(),  # Positions des 3 fourmis les plus proches (ant1_x, ant1_y, ...)
            1 if self.has_food else 0,  # Si la fourmi transporte de la nourriture
            distance_to_colony,  # Distance par rapport à la colonie
            *food_direction,
        ] + surrounding_pheromones.tolist()  # Ajouter les niveaux de phéromones par type

        #ic(self.inputs)

        return self.inputs



    def state(self):
        return self.inputs # size = 15
