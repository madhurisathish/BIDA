import numpy as np
from tqdm import tqdm
from utils import common

class AntColony:
    def __init__(self, file_path):
        self.node_coords = self.read_dataset(file_path)

    def read_dataset(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        node_coords = {}
        dimension = None

        for line in lines:
            if line.startswith('DIMENSION'):
                dimension = int(line.split(':')[1].strip())
            elif line.startswith('NODE_COORD_SECTION'):
                for _ in range(dimension):
                    data = lines[lines.index(line) + 1].split()
                    node_coords[int(data[0])] = tuple(map(int, data[1:]))
            elif line.startswith('DEPOT_SECTION'):
                break

        return node_coords
    

    def set_params(self, n_ants=10, n_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.n_customers = len(self.node_coords) - 1

        # Initialize pheromone trails
     

    def solve(self):
        best_solution = None
        best_fitness = float('-inf')

        for _ in tqdm(range(self.n_iterations)):
            solutions = []

            # Generate solutions for each ant
            for _ in range(self.n_ants):
                solution = self.generate_solution()
                solutions.append(solution)

            # Update pheromones
            self.update_pheromones(solutions)

            # Update best solution
            for solution in solutions:
                fitness = self.fitness(solution)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution

        return best_solution

    def generate_solution(self):
        solution = [0]  # Start at the depot
        unvisited_customers = set(range(1, self.n_customers + 1))

        while unvisited_customers:
            probabilities = self.calculate_probabilities(solution[-1], unvisited_customers)
            next_customer = np.random.choice(list(probabilities.keys()), p=list(probabilities.values()))
            solution.append(next_customer)
            unvisited_customers.remove(next_customer)

        solution.append(0)  # Return to depot
        return solution

    def calculate_probabilities(self, current_customer, unvisited_customers):
        probabilities = {}
        pheromone_row = self.pheromone[current_customer]
        total = 0

        for customer in unvisited_customers:
            pheromone = pheromone_row[customer]
            distance = self.distance_matrix[current_customer][customer]
            attractiveness = 1 / distance
            total += (pheromone ** self.alpha) * (attractiveness ** self.beta)

        for customer in unvisited_customers:
            pheromone = pheromone_row[customer]
            distance = self.distance_matrix[current_customer][customer]
            attractiveness = 1 / distance
            probabilities[customer] = ((pheromone ** self.alpha) * (attractiveness ** self.beta)) / total

        return probabilities

    def update_pheromones(self, solutions):
        evaporation = 1 - self.evaporation_rate
        for solution in solutions:
            total_fitness = self.fitness(solution)
            for i in range(len(solution) - 1):
                from_customer = solution[i]
                to_customer = solution[i + 1]
                self.pheromone[from_customer][to_customer] *= evaporation
                self.pheromone[from_customer][to_customer] += total_fitness

    def fitness(self, solution):
        return 1 / common.compute_route_length({'distance_matrix': self.distance_matrix}, solution)


# Example usage:
dataset_file = 'dataset.txt'  # Example dataset file path
aco = AntColony(dataset_file)
aco.set_params(n_ants=10, n_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5)
best_solution = aco.solve()
print("Best solution:", best_solution)
