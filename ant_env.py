import pygame
import numpy as np
import random
import neat


bg_color = (14, 255, 145)  # Couleur de fond
ant_initial_color = (0, 0, 0)  # Noir
circle_center = (60, 60)  # Centre de l'écran
circle_radius = 5
wall_color = (255, 255, 255)
death_zone_color = (150, 2, 55)
food_color = (255, 17, 150)


class PheromoneSystem:
    def __init__(self, width, height, num_types=3):
        self.width = width
        self.height = height
        self.num_types = num_types
        self.pheromones = np.zeros((width, height, num_types))
        self.evaporation_rate = 0.95  # 5% d'évaporation par étape

    def deposit(self, x, y, pheromone_type, amount):
        self.pheromones[x, y, pheromone_type] += amount

    def evaporate(self):
        self.pheromones *= self.evaporation_rate

    def get_pheromone_level(self, x, y, pheromone_type):
        return self.pheromones[x, y, pheromone_type]
    
    def draw_pheromones(self, screen):
        for x in range(self.width):
            for y in range(self.height):
                food_pheromone = self.pheromones[x, y, 0]
                path_pheromone = self.pheromones[x, y, 1]

                if food_pheromone > 0:
                    intensity = min(255, int(food_pheromone * 255))
                    pygame.draw.rect(screen, (255, 0, 0, intensity), (x, y, 1, 1))  # Rouge pour la nourriture

                if path_pheromone > 0:
                    intensity = min(255, int(path_pheromone * 255))
                    pygame.draw.rect(screen, (0, 0, 255, intensity), (x, y, 1, 1))  # Bleu pour le chemin



class Ant():
    def __init__(self, x, y, radius):
        self.pos = np.array([float(x), float(y)])  # Initialisation correcte des positions
        self.initial_pos = (400, 400)
        self.prev_pos = np.array([float(x), float(y)])
        self.has_food = False
        self.is_dead = False
        self.radius = radius  # Rayon de la fourmi (pour le cercle)
        self.circle = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
        self.color = (0,0,0)
        self.color2 = (255,0,0)
        self.update_vec_director = [0, 0]
        self.input = None
        self.output = None
        self.pheromone_type = None
        

    def move(self, env_size):
        random_x = random.uniform(-1, 1)
        random_y = random.uniform(-1, 1)
        vec_director = np.array([random_x, random_y])  # Création du vecteur directionnel
        self.update_vec_director = vec_director
        update_pos = self.pos + vec_director  # Ajout du vecteur position

        # Vérifier que la fourmi reste dans l'environnement
        if 0 <= update_pos[0] < env_size[0] and 0 <= update_pos[1] < env_size[1]:
            self.prev_pos = self.pos.copy()
            self.pos = update_pos
            self.circle.topleft = (self.pos[0] - self.radius, self.pos[1] - self.radius)

    def deposit_pheromone(self, pheromone_system):
        x, y = int(self.pos[0]), int(self.pos[1])
        if self.has_food:
            # Si la fourmi a trouvé de la nourriture, elle dépose un type spécifique de phéromone
            pheromone_system.deposit(x, y, pheromone_type=0, amount=1)  # Type 0 : Phéromone de nourriture
            self.pheromone_type = 0
            return self.pheromone_type
        else:
            # Si elle ne transporte pas de nourriture, elle peut déposer un autre type
            pheromone_system.deposit(x, y, pheromone_type=1, amount=0.5)  # Type 1 : Phéromone de chemin
            self.pheromone_type = 1
            return self.pheromone_type

    
    def get_inputs(self, pheromone_system, radius=5): # Le rayon a une grande importance sur la fluidité de la simulation, valeur élevé = cout computionnel élevé
        x, y = int(self.pos[0]), int(self.pos[1])
        
        # Niveaux de phéromones autour de la fourmi dans un rayon
        surrounding_pheromones_food = 0
        surrounding_pheromones_path = 0
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if 0 <= x + i < pheromone_system.width and 0 <= y + j < pheromone_system.height:
                    surrounding_pheromones_food += pheromone_system.get_pheromone_level(x + i, y + j, 0)
                    surrounding_pheromones_path += pheromone_system.get_pheromone_level(x + i, y + j, 1)

        self.input = [
            self.pos[0],  # Position x de la fourmi
            self.pos[1],  # Position y de la fourmi
            1 if self.has_food else 0,  # Booléen si elle possède de la nourriture
            surrounding_pheromones_food,  # Somme des phéromones de nourriture dans le voisinage
            surrounding_pheromones_path   # Somme des phéromones de chemin dans le voisinage
        ]
        
        return self.input

    def move_with_neat(self, output, env_size, pheromone_system, ants):
        if not self.is_dead:
            # Obtenir les niveaux de phéromones autour de la fourmi
            x, y = int(self.pos[0]), int(self.pos[1])
            food_pheromone_level = None
            path_pheromone_level = None
            if x and y < env_size[0]:
                food_pheromone_level = pheromone_system.get_pheromone_level(x, y, 0)  # Phéromones de nourriture
                path_pheromone_level = pheromone_system.get_pheromone_level(x, y, 1)  # Phéromones de chemin

            # Utiliser ces informations comme entrée du réseau neuronal
            self.inputs = [
                self.pos[0],  # Position x de la fourmi
                self.pos[1],  # Position y de la fourmi
                1 if self.has_food else 0,  # Booléen si elle possède de la nourriture
                food_pheromone_level,  # Niveau de phéromones de nourriture
                path_pheromone_level   # Niveau de phéromones de chemin
            ]

            # Les deux premières sorties sont le vecteur directionnel (dx, dy)
            dx = output[0]
            dy = output[1]
            
            vec_director = (dx, dy)

            update_pos = self.pos + vec_director  # Ajout du vecteur position

            # Vérifier que la fourmi reste dans l'environnement
            if 0 <= update_pos[0] < env_size[0] and 0 <= update_pos[1] < env_size[1]:
                self.pos = update_pos
                self.circle.topleft = (self.pos[0] - self.radius, self.pos[1] - self.radius)

            # Les deux autres sorties sont des booléens pour le dépôt de phéromones
            deposit_food_pheromone = output[2] > 0.5
            deposit_path_pheromone = output[3] > 0.5

            if deposit_food_pheromone:
                pheromone_system.deposit(int(self.pos[0]), int(self.pos[1]), 0, amount=1)
            if deposit_path_pheromone:
                pheromone_system.deposit(int(self.pos[0]), int(self.pos[1]), 1, amount=0.5)
    

    def draw(self, screen, color):
        # Dessine la fourmi
        if self.has_food == False:
            pygame.draw.circle(screen, color, (int(self.pos[0]), int(self.pos[1])), self.radius)
        else:
            pygame.draw.circle(screen, color, (int(self.pos[0]), int(self.pos[1])), self.radius)  


class Wall:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)  # Création du rectangle pour le mur

    def draw(self, screen):
        pygame.draw.rect(screen, wall_color, self.rect)  # Dessiner le mur


class Food:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)  # Création de la nourriture

    def draw(self, screen):
        pygame.draw.rect(screen, food_color, self.rect)  # Dessiner la nourriture


class DeathZone:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)  # Création du rectangle pour la zone

    def draw(self, screen):
        pygame.draw.rect(screen, death_zone_color, self.rect)  # Dessiner la zone


class Colony:
    def __init__(self, size, nb_ant, food_quantity, wall_number, death_zone_number):
        self.size = size
        self.cell_size = 10
        self.nb_ant = nb_ant
        self.ants = [Ant(size[0] // 2, size[1] // 2, self.cell_size/2) for _ in range(self.nb_ant)]  # Liste de fourmis
        self.occupied_positions = set()  # Ensemble pour suivre les positions occupées
        self.food = self.generate_food(food_quantity)
        self.wall = self.generate_wall(wall_number)
        self.death_zone = self.generate_death_zone(death_zone_number)
        self.pheromone_system = PheromoneSystem(width=size[0], height=size[1])

    def deposit_pheromones(self, ant):
        ant.deposit_pheromone(self.pheromone_system)

    def is_position_available(self, pos, width=1, height=1):
        """Vérifie si une zone rectangulaire est libre."""
        for x in range(pos[0], pos[0] + width):
            for y in range(pos[1], pos[1] + height):
                if (x, y) in self.occupied_positions:
                    return False
        return True

    def mark_position_as_occupied(self, pos, width=1, height=1):
        """Marque une zone rectangulaire comme occupée."""
        for x in range(pos[0], pos[0] + width):
            for y in range(pos[1], pos[1] + height):
                self.occupied_positions.add((x, y))

    def safe_zone(self):
        """Marque une zone sauve pour les fourmis"""
        safe_zone_size = 100
        safe_zone_x, safe_zone_y = self.size[0]/2, self.size[1]/2
        self.mark_position_as_occupied((safe_zone_x, safe_zone_y), safe_zone_size, safe_zone_size)


    def generate_food(self, quantity):
        """Créer un réctangle restreint aléatoire dans lequel la nourriture apparaîtra"""
        food = []

        # Taille de la sous-zone restreinte (20-50% de la taille de l'environnement)
        restricted_zone_width = random.randint(int(self.size[0] * 0.2), int(self.size[0] * 0.5))
        restricted_zone_height = random.randint(int(self.size[1] * 0.2), int(self.size[1] * 0.5))

        # Position aléatoire de la sous-zone dans l'aire de jeu
        restricted_zone_x = random.randint(0, self.size[0] - restricted_zone_width)
        restricted_zone_y = random.randint(0, self.size[1] - restricted_zone_height)

        # Générer la nourriture dans cette sous-zone restreinte
        for _ in range(quantity):
            while True:
                # Coordonnées aléatoires à l'intérieur de la sous-zone restreinte
                x = random.randint(restricted_zone_x, restricted_zone_x + restricted_zone_width - 1)
                y = random.randint(restricted_zone_y, restricted_zone_y + restricted_zone_height - 1)

                width = int(self.cell_size * 0.7)
                height = int(self.cell_size * 0.7)

                # Vérifier si la position est libre avant de placer la nourriture
                if self.is_position_available((x, y), width, height):
                    food.append(Food(x, y, width, height))
                    self.mark_position_as_occupied((x, y), width, height)
                    break

        return food


    def generate_wall(self, number):
        wall = []
        for _ in range(number):
            while True:
                x, y = random.randint(0, self.size[0] - 1), random.randint(0, self.size[1] - 1)
                width, height = random.randint(0, int(self.size[0] / 4)), random.randint(0, int(self.size[1] / 4))
                if self.is_position_available((x, y), width, height):
                    
                    wall.append(Wall(x, y, width, height))
                    self.mark_position_as_occupied((x, y), width, height)
                    break
        return wall

    def generate_death_zone(self, number):
        death_zone = []
        for _ in range(number):
            while True:
                x, y = random.randint(0, self.size[0] - 1), random.randint(0, self.size[1] - 1)
                zone_size = random.randint(0, int(self.size[0] / 10))
                if self.is_position_available((x, y), width=zone_size, height=zone_size):
                    death_zone.append(DeathZone(x, y, zone_size, zone_size))
                    self.mark_position_as_occupied((x, y), width=zone_size, height=zone_size)
                    break
        return death_zone

    def update(self):
        for ant in self.ants:
            if not ant.is_dead:
                ant.move(self.size)  # Met à jour la position de chaque fourmi
                self.deposit_pheromones(ant)  # Les fourmis déposent des phéromones pendant leur déplacement
        
        # Evaporation des phéromones à chaque cycle
        self.pheromone_system.evaporate()

    def neat_update(self, networks):
        # Utiliser une nouvelle liste pour stocker les fourmis et réseaux encore actifs
        new_ants = []
        new_networks = []

        for i, ant in enumerate(self.ants):
            if not ant.is_dead:
                if i < len(networks):
                    genome, net = networks[i]  # Récupérer le réseau neuronal et le génome associé à cette fourmi
                    inputs = ant.get_inputs(self.pheromone_system)  # Obtenir les entrées du réseau pour la fourmi
                    
                    output = net.activate(inputs)  # Activer le réseau neuronal
                    
                    ant.move_with_neat(output, self.size, self.pheromone_system, self.ants)

                    
                    # Si la fourmi est toujours en vie, la garder ainsi que son réseau neuronal
                    new_ants.append(ant)
                    new_networks.append((genome, net))

        # Remplacer l'ancienne liste par la nouvelle liste mise à jour
        self.ants = new_ants
        networks[:] = new_networks
                
        # Evaporation des phéromones à chaque cycle
        self.pheromone_system.evaporate()



class AntColonyPygame(Colony):
    def __init__(self, size, nb_ant, food_quantity, wall_number, death_zone_number):
        super().__init__(size, nb_ant, food_quantity, wall_number, death_zone_number)
        pygame.init()
        self.cell_size = 10
        self.screen = pygame.display.set_mode(size)
        self.wall_number = wall_number
        pygame.display.set_caption("Ant Colony Simulation")
        FONT_SIZE = 36
        # Définition de la police
        self.FONT_COLOR = (0, 0, 0)
        self.font = pygame.font.Font(None, FONT_SIZE)
        self.food_quantity = food_quantity


    def check_collisions_neat(self, networks):
        for ant in self.ants:
            for wall in self.wall:
                for zone in self.death_zone:
                    for food in self.food:

                        if ant.circle.colliderect(wall.rect):
                            
                            overlap_left = ant.circle.right - wall.rect.left
                            overlap_right = wall.rect.right - ant.circle.left
                            overlap_top = ant.circle.bottom - wall.rect.top
                            overlap_bottom = wall.rect.bottom - ant.circle.top
                            
                            # Trouver la direction de la plus petite pénétration
                            min_overlap = min(overlap_left, overlap_right, overlap_top, overlap_bottom)
                            
                            if min_overlap == overlap_left:
                                # Collision du côté gauche du mur
                                ant.pos[0] = wall.rect.left - ant.circle.width / 2
                                ant.update_vec_director[0] *= -1  # Inverser la direction horizontale
                                ant.update_vec_director += np.random.uniform(-0.2, 0.2, 2)
                            elif min_overlap == overlap_right:
                                # Collision du côté droit du mur
                                ant.pos[0] = wall.rect.right + ant.circle.width / 2
                                ant.update_vec_director[0] *= -1  # Inverser la direction horizontale
                                ant.update_vec_director += np.random.uniform(-0.2, 0.2, 2)
                            elif min_overlap == overlap_top:
                                # Collision avec le haut du mur
                                ant.pos[1] = wall.rect.top - ant.circle.height / 2
                                ant.update_vec_director[1] *= -1  # Inverser la direction verticale
                                ant.update_vec_director += np.random.uniform(-0.2, 0.2, 2)
                            elif min_overlap == overlap_bottom:
                                # Collision avec le bas du mur
                                ant.pos[1] = wall.rect.bottom + ant.circle.height / 2
                                ant.update_vec_director[1] *= -1  # Inverser la direction verticale
                                ant.update_vec_director += np.random.uniform(-0.2, 0.2, 2)
                            
                            # Ajuster la position de la fourmi
                            ant.circle.center = ant.pos
                            
                            # Ajouter un petit décalage aléatoire pour éviter les blocages
                            ant.update_vec_director += np.random.uniform(-0.1, 0.1, 2)
                            ant.update_vec_director = ant.update_vec_director / np.linalg.norm(ant.update_vec_director)
                            

                        elif ant.circle.colliderect(zone.rect):
                            ant.is_dead == True
                            if ant in self.ants:
                                self.ants.remove(ant)
                                for i, ant in enumerate(self.ants):
                                    g, net = networks[i]
                                    if g and net in networks:
                                        networks.remove(g)
                                        networks.remove(net)
                            
                             
                        elif ant.circle.colliderect(food.rect):
                            if ant.has_food == False:    
                                if food in self.food:
                                    self.food.remove(food)
                                ant.has_food = True
                            else:
                                pass

    def check_collisions(self):
        for ant in self.ants:
            for wall in self.wall:
                for zone in self.death_zone:
                    for food in self.food:

                        if ant.circle.colliderect(wall.rect):
                            
                            overlap_left = ant.circle.right - wall.rect.left
                            overlap_right = wall.rect.right - ant.circle.left
                            overlap_top = ant.circle.bottom - wall.rect.top
                            overlap_bottom = wall.rect.bottom - ant.circle.top
                            
                            # Trouver la direction de la plus petite pénétration
                            min_overlap = min(overlap_left, overlap_right, overlap_top, overlap_bottom)
                            
                            if min_overlap == overlap_left:
                                # Collision du côté gauche du mur
                                ant.pos[0] = wall.rect.left - ant.circle.width / 2
                                ant.update_vec_director[0] *= -1  # Inverser la direction horizontale
                                ant.update_vec_director += np.random.uniform(-0.2, 0.2, 2)
                            elif min_overlap == overlap_right:
                                # Collision du côté droit du mur
                                ant.pos[0] = wall.rect.right + ant.circle.width / 2
                                ant.update_vec_director[0] *= -1  # Inverser la direction horizontale
                                ant.update_vec_director += np.random.uniform(-0.2, 0.2, 2)
                            elif min_overlap == overlap_top:
                                # Collision avec le haut du mur
                                ant.pos[1] = wall.rect.top - ant.circle.height / 2
                                ant.update_vec_director[1] *= -1  # Inverser la direction verticale
                                ant.update_vec_director += np.random.uniform(-0.2, 0.2, 2)
                            elif min_overlap == overlap_bottom:
                                # Collision avec le bas du mur
                                ant.pos[1] = wall.rect.bottom + ant.circle.height / 2
                                ant.update_vec_director[1] *= -1  # Inverser la direction verticale
                                ant.update_vec_director += np.random.uniform(-0.2, 0.2, 2)
                            
                            # Ajuster la position de la fourmi
                            ant.circle.center = ant.pos
                            
                            # Ajouter un petit décalage aléatoire pour éviter les blocages
                            ant.update_vec_director += np.random.uniform(-0.1, 0.1, 2)
                            ant.update_vec_director = ant.update_vec_director / np.linalg.norm(ant.update_vec_director)
                            

                        elif ant.circle.colliderect(zone.rect):
                            ant.is_dead == True
                            if ant in self.ants:
                                self.ants.remove(ant)
                            
                            
                             
                        elif ant.circle.colliderect(food.rect):
                            if ant.has_food == False:    
                                if food in self.food:
                                    self.food.remove(food)
                                ant.has_food = True
                            else:
                                pass
                    

    def draw(self):
        self.screen.fill(bg_color)  # Fond vert

        # Dessiner la nourriture
        for food in self.food:
            food.draw(self.screen)

        # Dessiner les murs
        for wall in self.wall:
            wall.draw(self.screen)

        # Dessiner les zones mortelles
        for zone in self.death_zone:
            zone.draw(self.screen)

        # Dessiner les fourmis
        for ant in self.ants:
            if ant.has_food == True:
                color = ant.color2
            else:
                color = ant.color
            ant.draw(self.screen, color)

        #self.pheromone_system.draw_pheromones(self.screen)

        

    def run_simulation_with_display(self, steps, display):
        """
        Exécute la simulation avec ou sans affichage selon la valeur de 'display'.
        """

        if display:
            # Si display est activé, on affiche la simulation avec Pygame
            clock = pygame.time.Clock()
            for _ in range(steps):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                self.update()           # Mise à jour des fourmis
                self.check_collisions()  # Vérification des collisions
                self.draw()              # Dessin à l'écran
                clock.tick(60)           # Limitation à 30 FPS
        else:
            # Si display est désactivé, on ne fait qu'avancer la simulation sans l'afficher
            for _ in range(steps):
                self.update()
                self.check_collisions()
                # Aucune action graphique (draw, clock) car display est désactivé

    def run_simulation_with_neat(self, steps, display, net, generation):
        """
        Exécute la simulation avec ou sans affichage selon la valeur de 'display'.
        """
        
        if display:
            # Si display est activé, on affiche la simulation avec Pygame
            clock = pygame.time.Clock()
            for _ in range(steps):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                
                # Dessiner le reste de la simulation
                self.draw()              # Dessin à l'écran
                # Afficher le texte de la génération
                

                # Mettre à jour les fourmis et les collisions
                self.neat_update(net)           # Mise à jour des fourmis
                self.check_collisions_neat(net)  # Vérification des collisions

                generations = self.font.render(f"Generation: {generation}", True, self.FONT_COLOR)
                self.screen.blit(generations, (0, 0))

                nb_ant = self.font.render(f"Ants: {len(self.ants)}", True, self.FONT_COLOR)
                self.screen.blit(nb_ant, (0, 30))

                fruit = self.font.render(f"Fruit: {sum(1 for ant in self.ants if ant.has_food)}/{self.food_quantity}", True, self.FONT_COLOR)
                self.screen.blit(fruit, (0, 60))

                # Mettre à jour l'affichage pour montrer les changements (incluant le compteur)
                pygame.display.flip()

                clock.tick(60)  # Limitation à 240 FPS (ou change à ce que tu veux)

        else:
            # Si display est désactivé, on ne fait qu'avancer la simulation sans l'afficher
            for _ in range(steps):
                self.neat_update(net) 
                self.check_collisions_neat(net)
                # Aucune action graphique (draw, clock) car display est désactivé
            


    def stop_and_restart_simulation(self, display):
        """
        Arrête la simulation et la redémarre en réinitialisant les fourmis et autres éléments.
        """
        # Réinitialiser la position des fourmis (par exemple, toutes au centre)
        for ant in self.ants:
            ant.pos = np.array([self.size[0] // 2, self.size[1] // 2])
            ant.circle.topleft = (ant.pos[0] - ant.radius, ant.pos[1] - ant.radius)  # Réinitialisation du rectangle associé

        # Si tu veux également réinitialiser ou régénérer les murs, tu peux le faire ici :
        #self.walls = self.generate_wall(self.wall_number)

        # Re-lancer la simulation
        print("Restarting simulation...")
        self.run_simulation_with_display(steps=10000, display=display)


# Initialisation de la simulation
env_size = (800, 800)
generations = 500
config_path = 'config.txt'
steps = 600
# Configuration NEAT

def compute_total_food_collected(ants):
    """
    Calcule la quantité totale de nourriture rapportée à la colonie.
    
    :param ants: La liste des fourmis dans la colonie.
    :return: Le nombre total de nourriture collectée par la colonie.
    """
    total_food_collected = sum(1 for ant in ants if ant.has_food)
    return total_food_collected


def fitness_function(genomes, ants):
    """
    Évalue la performance de la colonie en fonction du nombre total de nourriture rapportée.
    
    :param genomes: La liste des génomes à évaluer.
    :param ants: La liste des fourmis.
    """
    # Calculer la nourriture totale collectée par la colonie
    total_food_collected = compute_total_food_collected(ants)

    # Distribuer une partie du fitness basé sur la contribution collective
    for i, ant in enumerate(ants):
        genome = genomes[i][1]
        fitness = 0
        
        # Récompenser en fonction de la nourriture totale collectée par la colonie
        fitness += total_food_collected * 10  # Récompense collective pour la colonie

        # Optionnel : Récompenser les fourmis individuellement pour d'autres comportements
        if ant.has_food:
            fitness += 5  # Récompenser un peu la collecte individuelle

        # Pénaliser les fourmis immobiles (si la position n'a pas changé)
        if np.array_equal(ant.pos, ant.prev_pos):
            fitness -= 5  # Pénaliser la fourmi si elle n'a pas bougé depuis la dernière étape
        
        # Vérifier si le type de phéromone déposé correspond à l'état de la fourmi
        # Si la fourmi a de la nourriture, elle doit déposer une phéromone de nourriture
        if ant.deposit_pheromone:
            if ant.has_food and ant.pheromone_type != 0:
                fitness -= 5  # Pénaliser si elle a de la nourriture mais dépose une mauvaise phéromone
            elif not ant.has_food and ant.pheromone_type != 1:
                fitness -= 5  # Pénaliser si elle n'a pas de nourriture mais dépose une phéromone de nourriture
            else:
                fitness += 2  # Récompenser si elle dépose la bonne phéromone pour son état

        # Pénaliser les fourmis mortes
        if ant.is_dead:
            fitness -= 10  # Pénalisation pour la mort
        
        # Affecter le fitness au génome correspondant
        genome.fitness = fitness


counter = 0

def eval_genomes(genomes, config):
    # Créer une liste pour stocker les réseaux neuronaux associés aux fourmis
    networks = []
    global counter
    ant_colony = AntColonyPygame(size=env_size, nb_ant=100, food_quantity=15, wall_number=1, death_zone_number=1)

    # Associer chaque fourmi à un génome et un réseau neuronal
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0
        networks.append((genome, net))  # Ajouter le réseau neuronal et le génome associés à la fourmi

    # Vérifier que le nombre de fourmis correspond bien au nombre de réseaux neuronaux
        #if len(networks) != len(ant_colony.ants):
        #print(len(networks))
        #print(len(ant_colony.ants))
    #raise ValueError("Le nombre de réseaux neuronaux ne correspond pas au nombre de fourmis.")

   
    # Lancer la simulation pour toutes les fourmis avec les réseaux neuronaux
    ant_colony.run_simulation_with_neat(steps=steps, display=True, net=networks, generation=counter)

    # Calculer la fitness de chaque fourmi
    fitness_function(genomes, ant_colony.ants)
    counter += 1



def run(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(eval_genomes, generations)

run(config_path)
"""
for i in range(1):
    print(f"Episode {i+1}")
    ant_colony = AntColonyPygame(size=env_size, nb_ant=100, food_quantity=15, wall_number=50, death_zone_number=1)
    ant_colony.run_simulation_with_display(steps=1000, display=True)
"""