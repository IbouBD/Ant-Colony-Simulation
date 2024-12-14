# Ant Colony Simulation

Simulating an ant colony using **NEAT (NeuroEvolution of Augmenting Topologies)** and a **grid-based environment**. This project demonstrates how artificial ants can evolve behaviors such as collecting fruits, navigating obstacles, and collaborating using pheromone trails.

---

## Features

### **1. Simulated Ant Colony Behavior**
- **Foraging and Collaboration:**
  - Ants explore the environment to find fruits.
  - Use pheromones to leave trails for other ants.
  - Collaborate to efficiently bring fruits back to the colony.

- **Dynamic Environment:**
  - Grid-based simulation with walls, death zones, fruits, and pheromones.
  - Pheromones evaporate over time to simulate natural decay.

### **2. NeuroEvolution with NEAT**
- Each ant is controlled by a neural network evolved using the **NEAT** algorithm.
- Networks adapt over generations to optimize ant behavior.
- Neural network inputs:
  - Position of the ant.
  - Local pheromone levels (per type).
  - Whether the ant is carrying food or not.
- Neural network outputs:
  - Movement direction (`dx`, `dy`).
  - Decision to deposit pheromones.
  - Choice of pheromone type.

### **3. Visualization**
- Real-time simulation using **Pygame** to visualize:
  - Ant movements and actions.
  - Pheromone trails with intensity levels.
  - Fruits, walls, and death zones.

---

## Requirements

### **Python Libraries**
- Python 3.7+
- [NumPy](https://numpy.org/)
- [Pygame](https://www.pygame.org/)
- [NEAT-Python](https://neat-python.readthedocs.io/)

To install the dependencies:
```bash
pip install numpy pygame neat-python
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Ant-Colony-Simulation.git
   cd Ant-Colony-Simulation
   ```

2. Ensure you have Python 3.7 or higher installed.

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## How It Works

### **Core Components**

1. **Ants (`Ant` class):**
   - Each ant is an agent controlled by a neural network.
   - They perceive their environment, make decisions, and act accordingly.

2. **Environment (`Environment` class):**
   - A grid-based representation of the simulation space.
   - Includes fruits, walls, death zones, and pheromones.

3. **Pheromone System (`PheromoneSystem` class):**
   - A grid storing pheromone levels by type.
   - Handles pheromone deposition and evaporation.

4. **NeuroEvolution (`NEAT`):**
   - Ant behaviors are evolved using the NEAT algorithm.
   - Fitness is based on tasks like collecting fruits and collaborating effectively.

### **Simulation Workflow**
1. **Initialize Environment:**
   - Set up the grid with fruits, walls, and death zones.
   - Spawn ants at the colony.

2. **Run Generations:**
   - Each generation:
     - Evaluate ant behaviors.
     - Adjust neural networks using NEAT.

3. **Visualization:**
   - Observe ant movements, pheromones, and environment interactions in real-time.

---

## Running the Simulation

### **1. Training the Colony**
To train the ants using NEAT:
```bash
python train.py
```
- This script will run the simulation over multiple generations.
- Results (best genomes) are saved for later replay.

### **2. Replaying the Best Genome**
To replay the behavior of the best ant genome:
```bash
python replay.py
```
- Visualize the actions of the most successful ants evolved during training.

---

## Configuration

### **NEAT Configuration**
The NEAT algorithm parameters are defined in `config.txt`. Example:
```ini
[NEAT]
pop_size              = 50
fitness_criterion     = max
fitness_threshold     = 200
reset_on_extinction   = True
...
```

### **Fitness Function**
The fitness function rewards ants for:
- Collecting fruits.
- Returning fruits to the colony.
- Exploring efficiently.
- Avoiding obstacles and death zones.

Pénalités sont appliquées pour:
- Mourir dans une zone mortelle.
- Comportements inutiles ou inefficaces.

---

## Folder Structure
```
Ant-Colony-Simulation/
├── train.py             # Script for training the ants with NEAT
├── replay.py            # Script for replaying the best genome
├── simulation.py        # Core simulation logic
├── colony.py            # Definitions for ants and colony behavior
├── config.txt           # NEAT configuration file
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## Future Enhancements
- Introduce specialization (e.g., scouts vs collectors).
- Add dynamic obstacles and real-time environmental changes.
- Simulate competition between multiple colonies.
- Improve pheromone modeling with diffusion and decay.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
- [NEAT-Python](https://neat-python.readthedocs.io/) for enabling neuroevolution.
- Inspiration from biological ant colony behaviors.

---

## Contributing
Feel free to fork this repository, submit issues, or create pull requests to improve the simulation. Collaboration is welcome!

