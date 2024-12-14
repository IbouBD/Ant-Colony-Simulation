import numpy as np
import pygame
from colony import Ant
import random
import neat
import pickle

CELL_SIZE = 10  # Taille d'une cellule en pixels

# Couleurs
COLOR_EMPTY = (14, 255, 145)  # Vert pour les cases vides
COLOR_WALL = (255, 255, 255)  # Blanc pour les murs
COLOR_DEATH_ZONE = (255, 0, 0)  # Rouge pour les zones mortelles
COLOR_FOOD = (0, 0, 255)  # Bleu pour la nourriture
COLOR_ANT = (0, 0, 0)  # Noir pour les fourmis 
COLOR_PHEROMONE = (128, 0, 128) # Violet pour les phéromones

pygame.init()
pygame.display.set_caption("Ant Colony Simulation with NEAT")
clock = pygame.time.Clock()

# Définisson le système de phéromones :
class PheromoneSystem:
    def __init__(self, width, height, num_types=3):
        # Dimensions adaptées à NumPy : [row, column, pheromone_types]
        self.width = width
        self.height = height
        self.num_types = num_types
        self.pheromones = np.zeros((height, width, num_types))  # [y, x, types]
        self.evaporation_rate = 0.95  # 5% d'évaporation par étape

    def deposit(self, x, y, pheromone_type, amount):
        """Ajoute une quantité de phéromones d'un certain type à une position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pheromones[y, x, pheromone_type] += amount  # Indices corrigés [y, x]

    def evaporate(self):
        """Applique l'évaporation des phéromones sur toute la grille."""
        self.pheromones *= self.evaporation_rate

    def get_pheromone_level(self, x, y, pheromone_type):
        """Récupère le niveau de phéromones d'un certain type à une position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.pheromones[y, x, pheromone_type]  # Indices corrigés [y, x]
        return 0


# Définission de l'environnement sous forme de grille numpy
class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid_size = (width, height)
        self.grid = np.zeros(self.grid_size, dtype=np.int8)  # Grille principale
        self.walls = np.zeros(self.grid_size, dtype=np.int8)  # Grille binaire pour les murs
        self.death_zones = np.zeros(self.grid_size, dtype=np.int8)  # Grille binaire pour zones mortelles
        self.food = np.zeros(self.grid_size, dtype=np.int8)  # Grille binaire pour nourriture
        self.pheromones = np.zeros((*self.grid_size, 2), dtype=np.float32)  # Grille 3D pour phéromones (2 types)
        self.food = np.zeros(self.grid_size, dtype=np.int8)  # Grille binaire pour nourriture
        self.ants = []
        self.pheromone_system = PheromoneSystem(width, height)
        self.occupied_positions = set()  # Ensemble pour suivre les positions occupées

    def is_position_available(self, pos, width=1, height=1):
        x, y = pos
        if 0 <= x < self.width and 0 <= y < self.height:
            return np.all(self.grid[y:y+height, x:x+width] == 0)
        return False


    def mark_position_as_occupied(self, pos = tuple, width=1, height=1):
        x, y = pos
        self.grid[y:y+height, x:x+width] = 2  # 2 représente la nourriture


    def safe_zone(self):
        """Marque une zone sauve pour les fourmis"""
        safe_zone_size = 100
        safe_zone_x, safe_zone_y = self.width/2, self.height[1]/2
        self.mark_position_as_occupied((safe_zone_x, safe_zone_y), safe_zone_size, safe_zone_size)

    def add_wall(self, x, y, width, height):
        """Ajoute un mur à la grille."""
        self.walls[y:y+height, x:x+width] = 1
        self.grid[y:y+height, x:x+width] = 1  # Met à jour la grille principale


    def add_food(self, x, y, width, height):
        """Ajoute de la nourriture à la grille."""
        self.food[y:y+height, x:x+width] = 1
        self.grid[y:y+height, x:x+width] = 2  # Met à jour la grille principale


    def generate_food(self, quantity):
        """Créer un rectangle restreint aléatoire dans lequel la nourriture apparaîtra."""

        # Taille de la sous-zone restreinte (20-50% de la taille de l'environnement)
        restricted_zone_width = random.randint(int(self.width * 0.2), int(self.width * 0.5))
        restricted_zone_height = random.randint(int(self.height * 0.2), int(self.height * 0.5))

        # Position aléatoire de la sous-zone dans l'aire de jeu
        restricted_zone_x = random.randint(0, self.width - restricted_zone_width)
        restricted_zone_y = random.randint(0, self.height - restricted_zone_height)

        #restricted_zone_x = 1
        #restricted_zone_y = 30

        # Générer la nourriture dans cette sous-zone restreinte
        for _ in range(quantity):
            while True:
                # Coordonnées aléatoires à l'intérieur de la sous-zone restreinte
                x = random.randint(restricted_zone_x, restricted_zone_x + restricted_zone_width - 1)
                y = random.randint(restricted_zone_y, restricted_zone_y + restricted_zone_height - 1)

                # Vérifier si la position est libre avant de placer la nourriture
                if self.is_position_available((y, x)):
                    self.add_food(y, x, 1, 1)
                    self.mark_position_as_occupied((y, x))
                    break



    def add_death_zone(self, x, y, width, height):
        """Ajoute une zone mortelle à la grille."""
        self.death_zones[y:y+height, x:x+width] = 1
        self.grid[y:y+height, x:x+width] = 3  # Met à jour la grille principale

    def add_ant(self, ant):
        """Ajoute une fourmi à une position spécifique."""
        self.ants.append((ant.pos))  # Ajouter la position de la fourmi
        x, y = int(ant.pos[0]), int(ant.pos[1])
        self.grid[x, y] = 4  # Marquer la fourmi sur la grille principale
        pass


    def check_collisions(self, ant):
        """
        Gère les collisions entre les fourmis, les murs, les zones mortelles et la nourriture.
        
        Args:
            ant: L'objet fourmi dont on vérifie les collisions
        """
        # Arrondir les coordonnées à l'entier le plus proche
        x, y = round(ant.pos[0]), round(ant.pos[1])
        
        # Vérification des limites de la grille
        if 0 <= x < self.width and 0 <= y < self.height:
            # Vérification des collisions avec les murs
            if self.walls[y][x] == 1:
                self.handle_wall_collision(ant)
                return  # Sortir après une collision avec un mur
                
            # Vérification des collisions avec les zones mortelles
            if self.death_zones[y][x] == 1:
                self.handle_death_zone_collision(ant)
                return  # Sortir après une collision mortelle
                
            # Vérification des collisions avec la nourriture
            if self.food[y][x] == 1 and not ant.has_food:
                ant.has_food = True
                self.handle_food_collision(ant, (y, x))
        else:
            # Gestion des positions hors limites
            self.handle_wall_collision(ant)

    def handle_wall_collision(self, ant):
        """
        Gère la collision entre une fourmi et un mur.
        """
        # Inverser la direction du mouvement
        #ant.update_vec_director *= -1

        # Ajouter un petit décalage aléatoire pour éviter les blocages
        #ant.update_vec_director += np.random.uniform(-0.1, 0.1, 2)
        #ant.update_vec_director /= np.linalg.norm(ant.update_vec_director)

        # Reculer légèrement pour éviter le mur
        ant.pos = np.clip(ant.pos, [0, 0], [self.width - 1, self.height - 1])

    def handle_death_zone_collision(self, ant):
        """
        Gère la collision entre une fourmi et une zone mortelle.
        """
        ant.is_dead = True
        x, y = int(ant.pos[0]), int(ant.pos[1])
        self.grid[y, x] = 0

    def handle_food_collision(self, ant, food_pos):
        """
        Gère la collision entre une fourmi et de la nourriture.
        """
        if ant.has_food == False:
            ant.has_food = True
            x, y = food_pos
            self.food[y, x] = 0
            self.grid[y, x] = 0  # Retirer la nourriture de la grille
            ant.pickup_location(x, y)
        else:
            pass

    def move_ant(self, ant, new_x, new_y):
        """
        Déplace une fourmi à une nouvelle position.
        
        Args:
            ant: L'objet fourmi à déplacer
            new_x (float): Le déplacement en x à effectuer
            new_y (float): Le déplacement en y à effectuer
        """
        if ant.is_dead:
            return
            
        # Obtenir et sauvegarder la position actuelle
        current_x, current_y = round(ant.pos[0]), round(ant.pos[1])
        ant.old_pos = np.array([current_x, current_y])
        
        # Calculer la nouvelle position
        new_pos = ant.pos + np.array([new_x, new_y])
        rounded_x, rounded_y = round(new_pos[0]), round(new_pos[1])
        
        # Vérifier les limites de la grille
        if 0 <= rounded_x < self.width and 0 <= rounded_y < self.height:
            # Effacer l'ancienne position dans la grille
            self.grid[current_y][current_x] = 0
            # Mettre à jour la position de la fourmi
            ant.pos = new_pos
            # Mettre à jour la grille avec la nouvelle position
            self.grid[rounded_y][rounded_x] = 4  # Utiliser 1 au lieu de 4

            


    def display_grid(self):
        """Affiche la grille avec des symboles pour chaque entité."""
        
        symbols = {0: '.', 1: '#', 2: 'F', 3: 'D', 4: 'A'}  # Symboles pour vide, mur, nourriture, zone mortelle, fourmi
        for y in range(self.height):
            row = ''.join(symbols[self.grid[y, x]] for x in range(self.width))
            #print(row)
        
        print(self.grid)

    def draw_grid(self, screen):
        """
        Dessine la grille en fonction de son contenu.
        :param screen: Surface Pygame où dessiner.
        :param environment: Instance de l'environnement contenant les grilles.
        """
        for x in range(self.width):
            for y in range(self.height):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)


                # Déterminer la couleur de la cellule
                if self.grid[y, x] == 1:
                    color = COLOR_WALL
                elif self.grid[y, x] == 3:
                    color = COLOR_DEATH_ZONE
                elif self.grid[y, x] == 2:
                    color = COLOR_FOOD
                elif self.grid[y, x] == 4:
                    color = COLOR_ANT
                else:
                    color = COLOR_EMPTY

                # Dessiner la cellule
                pygame.draw.rect(screen, color, rect)

                # Dessiner les phéromones en superposition
                for pheromone_type in range(self.pheromone_system.num_types):
                    pheromone_level = self.pheromone_system.get_pheromone_level(x, y, pheromone_type)
                    if pheromone_level > 0:  # Si un niveau de phéromone existe
                        # Ajuster l'intensité de la couleur en fonction du niveau
                        alpha = min(255, int(pheromone_level * 50))  # Convertir le niveau en transparence (0-255)
                        if pheromone_type == 0:
                            pheromone_color = (alpha, 0, alpha)  # violet translucide pour les phéromones de type 0
                        if pheromone_type == 1:
                            pheromone_color = (alpha, 0, 0)  # rouge translucide pour les phéromones de type 1
                        if pheromone_type == 2:
                            pheromone_color = (0, alpha, alpha)  # bleu translucide pour les phéromones de type 2
                        if pheromone_color == (0, 0, 0):
                            pheromone_color = COLOR_EMPTY
                        pygame.draw.rect(screen, pheromone_color, rect, width=1)  # Superposition fine


class Simulation:
    def __init__(self, width = int, height = int, num_ants = int, num_food = int):
        self.env = Environment(width, height)
        self.colony_pos = (width // 2, height // 2)
        self.env.mark_position_as_occupied(self.colony_pos)
        self.ants = [Ant(self.colony_pos[0], self.colony_pos[1], self.env) for _ in range(num_ants)]
        self.env.generate_food(num_food)
        self.num_food = num_food
        self.total_food_collected = self.compute_total_food_collected(self.ants)
        self.max_idle_time = 150 # Exemple de temps max d'inactivité

    def compute_total_food_collected(self, ants): 
        return sum(ant.has_food for ant in ants) 
    
    def compute_distance(self, pos1, pos2): 
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def compute_total_food_collected(self, ants):
        """
        Calcule la quantité totale de nourriture rapportée à la colonie.
        
        :param ants: La liste des fourmis dans la colonie.
        :return: Le nombre total de nourriture collectée par la colonie.
        """
        total_food_collected = sum(1 for ant in ants if ant.has_food)
        return total_food_collected
    
    def compute_fitness(self, ant):
        """
        Calcule la fitness d'une fourmi pour encourager la collecte de tous les fruits.

        :param ant: L'objet fourmi.
        :return: Fitness calculée pour cette fourmi.
        """
        fitness = 0

        # Récompense individuelle pour ramasser un fruit
        if ant.has_food:
            fitness += 3  # Encourager la collecte individuelle

        # Pénalité pour ne pas déposer des phéromones
        if not ant.deposit_pheromone:
            fitness -= 3  # Pénaliser la non-déposition de phérom

        # Récompense supplémentaire pour ramener un fruit à la colonie
        x, y = int(ant.pos[0]), int(ant.pos[1])
        if (x, y) == self.colony_pos and ant.has_food:
            fitness += 4  # Récompense élevée pour un retour réussi
            ant.has_food = False  # Le fruit est déposé à la colonie

        # Récompense collective proportionnelle au progrès global
        if self.total_food_collected > 0:
            progress_ratio = self.total_food_collected / self.num_food
            fitness += 1.5 * progress_ratio  # Encourage la collaboration

        # Pénalité pour mort
        if ant.is_dead:
            fitness -= 1  # Pénalité pour mort

        # Bonus pour distance parcourue avec un fruit (encourage l'exploration efficace)
        if ant.has_food:
            distance_traveled = self.compute_distance(ant.pos, self.colony_pos)
            fitness += distance_traveled * 0.01  # Récompense pour l'exploration efficace

        # Pénalité pour comportements inefficaces (par exemple, trop de temps sans ramasser de fruit)
        if not ant.has_food and ant.idle_time > self.max_idle_time:
            fitness -= 0.5

        # Limiter la fitness maximale à 10
        if fitness > 10:
            fitness = 10

        return fitness



    def update(self, networks):
        """Met à jour l'état de la simulation à chaque étape."""
        for i, ant in enumerate(self.ants):
            if not ant.is_dead:
                if i < len(networks):
                    genome, net = networks[i]
                    self.update_ant(ant, genome, net)  # Gère le déplacement et les phéromones

        # Appliquer l'évaporation des phéromones
        self.env.pheromone_system.evaporate()

    def update_ant(self, ant, genome, net):
        """Met à jour une seule fourmi."""
        x = ant.get_inputs(self.env)
        output = net.activate(x)
        # Déplacer la fourmi de manière aléatoire
        #rx, ry = random.randint(-1, 1), random.randint(-1, 1)
        # Déplacer la fourmi et déposer des phéromones en utilisant les outputs du réseau neuronal
        # outupt size = 4
        dx = int(output[0]) # mvt en x
        dy = int(output[1]) # mvt en y
        z_type = int(output[2]) # type de phéromones
        z_amount = int(output[3]) * 10 # quantitées de pheromones lachés multiplié par un facteur 10
        self.env.move_ant(ant, dx, dy)

        if z_amount != 0:
            ant.deposit_pheromone = True
            ant.type = z_type
        else:
            ant.deposit_pheromone = False
            ant.type = z_type

        x, y = int(ant.pos[0]), int(ant.pos[1])
        # Dépôt de phéromones avec une probabilité de 25%
        #if 0 <= x < self.env.width and 0 <= y < self.env.height and random.random() < 0.25:
            #self.env.pheromone_system.deposit(x, y, pheromone_type=0, amount=10)
        self.env.pheromone_system.deposit(x, y, pheromone_type=z_type, amount=z_amount)

        # Gérer les collisions après le déplacement
        self.env.check_collisions(ant)
        # Affecter le fitness au génome
        fitness = self.compute_fitness(ant)
        genome.fitness += fitness

        # Afficher l'état de la fourmi pour débogage
        #print(f" Inputs:{ant.state()}")
        #print()
        #print(f"Output : {output}")
        
    def run(self, steps, display, networks):
        """Lance la simulation pour un nombre donné de pas."""
        
        if display == False:
            for _ in range(steps):
                self.update(networks)
            #self.env.display_grid()
        else:
            # Boucle principale
            screen = pygame.display.set_mode((self.env.width * CELL_SIZE, self.env.height * CELL_SIZE))

            clock = pygame.time.Clock()
            
            for _ in range(steps):
                    # Mettre à jour l'état des fourmis
                    self.update(networks)
                    # Effacer l'écran
                    screen.fill(COLOR_EMPTY)

                    # Dessiner la grille et les fourmis
                    self.env.draw_grid(screen)

                    # Mettre à jour l'affichage
                    pygame.display.flip()
                    clock.tick(60)  # Limiter à 60 FPS

            pygame.quit()
            self.colony_pos


steps = 150
config_path = 'config.txt'
generations = 10000
pop_size = 50
display = True # En 'True' on teste la population. En 'False' on entraine la population
env_size = 50
# temps < 0.5 seconds pour 1 générations avec une population de 50 fourmis

# Charger le meilleur génome sauvegardé
with open('best_genome.pkl', 'rb') as f:
    best_genome = pickle.load(f)

# Charger la population
with open('population.pkl', 'rb') as f:
    population = pickle.load(f)

def eval_genomes(genomes, config):
    # Création de l'environnement
    env = Simulation(width=env_size, height=env_size, num_ants=pop_size, num_food=12)
    # Créer une liste pour stocker les réseaux neuronaux associés aux fourmis
    networks = []
    

    # Associer chaque fourmi à un génome et un réseau neuronal
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0
        networks.append((genome, net))  # Ajouter le réseau neuronal et le génome associés à la fourmi


    # Lancer la simulation pour toutes les fourmis avec les réseaux neuronaux
    env.run(steps=steps, display=display, networks=networks)


def run(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.population = population

    try:
        winner = p.run(eval_genomes, generations)
    except neat.CompleteExtinctionException:
        print("Extinction complète : réinitialisation de la population.")
        p = neat.Population(config)  # Réinitialiser la population avec la même configuration
        winner = p.run(eval_genomes, generations)

    with open('best_gen.pkl', 'wb') as f:
        pickle.dump(winner, f)

    with open('pop.pkl', 'wb') as f:
        pickle.dump(p.population, f)


def test(genomes, config_path):
    """
    Fonction pour tester les réseaux neuronaux générés par la génération précédente en chargeant le meilleur génome.

    Args:
        genomes (list): Liste de tuples (id, génome).
        config_path (str): Chemin vers le fichier de configuration.
    """
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    population = neat.Population(config)
    
    # Charger le génome passé en argument
    for genome_id, genome in genomes:
        genome_id = 0  # ID de la population pour le meilleur génome
        population.population[genome_id] = genome

    # Ajouter des reporters pour visualiser les résultats
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    try:
        winner = population.run(eval_genomes, 1)  # Exécuter pour une seule génération
    except neat.CompleteExtinctionException:
        print("Extinction complète : réinitialisation de la population.")
        population = neat.Population(config)  # Réinitialiser la population avec la même configuration
        winner = population.run(eval_genomes, 1)  # Exécuter pour une seule génération

def replay_genome(config_path='config.txt', genome_path="best_gen.pkl"):
    """
    Fonction pour rejouer un génome sauvegardé.

    Args:
        config_path (str): Chemin vers le fichier de configuration.
        genome_path (str): Chemin vers le fichier contenant le génome sauvegardé.
    """
    # Charger le génome sauvegardé
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    # Convertir le génome chargé en une structure de données requise
    genomes = [(1, genome)]

    # Appeler la fonction de test avec le génome chargé
    test(genomes, config_path)


if display == False:
    # Entrainement de la population
    run(config_path)
else:
    # Tester le meilleur genome
    replay_genome()

# faire attention au surapprentissage, 