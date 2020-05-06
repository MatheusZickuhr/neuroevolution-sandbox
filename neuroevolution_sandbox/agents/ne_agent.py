import numpy as np

from python_ne.core.ga.ga_neural_network import GaNeuralNetwork
from python_ne.core.ga.genetic_algorithm import GeneticAlgorithm
from python_ne.core.neural_network import normalizer


class NeAgent:

    def __init__(self, env_adapter, model_adapter):
        self.env_adapter = env_adapter
        self.model_adapter = model_adapter
        self.best_element = None
        self.genetic_algorithm = None
        self.play_n_times = 1
        self.max_n_steps = float('inf')
        self.reward_if_max_step_reached = 0

    def train(self, number_of_generations, selection_percentage, mutation_chance,
              population_size, fitness_threshold, neural_network_config, crossover_strategy,
              mutation_strategy, play_n_times=1, max_n_steps=float('inf'),
              reward_if_max_step_reached=0, loggers=()):

        self.play_n_times = play_n_times
        self.max_n_steps = max_n_steps
        self.reward_if_max_step_reached = reward_if_max_step_reached

        self.genetic_algorithm = GeneticAlgorithm(
            population_size=population_size,
            selection_percentage=selection_percentage,
            mutation_chance=mutation_chance,
            fitness_threshold=fitness_threshold,
            mutation_strategy=mutation_strategy,
            crossover_strategy=crossover_strategy,
            model_adapter=self.model_adapter,
            neural_network_config=neural_network_config
        )

        for logger in loggers:
            self.genetic_algorithm.add_observer(logger)

        try:
            self.genetic_algorithm.run(
                number_of_generations=number_of_generations,
                calculate_fitness_callback=self.calculate_fitness
            )
        except KeyboardInterrupt:
            print('training canceled')

        self.best_element = self.genetic_algorithm.get_best_element()

    def calculate_fitness(self, element):
        return np.mean([self.play(element) for _ in range(self.play_n_times)])

    def save(self, file_path):
        self.best_element.save(file_path)

    def load(self, file_path):
        self.best_element = GaNeuralNetwork(
            create_model=False,
            model_adapter=self.model_adapter
        )
        self.best_element.load(file_path)

    def play(self, element=None):
        element = self.best_element if element is None else element
        self.env_adapter.reset()
        done = False
        observation, _, _ = self.env_adapter.step(self.env_adapter.get_random_action())
        fitness = 1
        step_count = 0
        while not done and step_count <= self.max_n_steps:
            observation = normalizer.normalize(observation)

            if self.env_adapter.is_continuous():
                action = element.get_output(np.array(observation))
            else:
                action = np.argmax(element.get_output(np.array(observation)))

            observation, reward, done = self.env_adapter.step(action)
            fitness += reward
            step_count += 1

        return fitness + self.reward_if_max_step_reached if step_count == self.max_n_steps else fitness
