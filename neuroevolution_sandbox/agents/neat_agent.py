import numpy as np

import neat


class NeatAgent:

    def __init__(self, env_adapter, config_file_path):
        self.env_adapter = env_adapter
        self.play_n_times = 1
        self.max_n_steps = float('inf')
        self.reward_if_max_step_reached = 0

        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file_path
        )

        self.best_net = None

    def train(self, number_of_generations, play_n_times=1, max_n_steps=float('inf'),
              reward_if_max_step_reached=0, reporters=()):

        self.play_n_times = play_n_times
        self.max_n_steps = max_n_steps
        self.reward_if_max_step_reached = reward_if_max_step_reached

        p = neat.Population(self.config)

        for reporter in reporters:
            p.add_reporter(reporter)

        winner = p.run(self.calculate_fitness, number_of_generations)
        self.best_net = neat.nn.FeedForwardNetwork.create(winner, self.config)

    def calculate_fitness(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = np.mean([self.play(net) for _ in range(self.play_n_times)])

    def save(self, file_path):
        pass

    def load(self, file_path):
        p = neat.Checkpointer.restore_checkpoint(file_path)
        filtered_population = [e for e in list(p.population.values()) if e.fitness is not None]
        winner = max(filtered_population, key=lambda e: e.fitness)
        self.best_net = neat.nn.FeedForwardNetwork.create(winner, self.config)

    def play(self, net=None):
        net = self.best_net if net is None else net
        self.env_adapter.reset()
        done = False
        observation, _, _ = self.env_adapter.step(self.env_adapter.get_random_action())
        fitness = 1
        step_count = 0
        while not done and step_count <= self.max_n_steps:

            if self.env_adapter.is_continuous():
                action = net.activate(np.array(observation))
            else:
                action = np.argmax(net.activate(np.array(observation)))

            observation, reward, done = self.env_adapter.step(action)
            fitness += reward
            step_count += 1

        return fitness + self.reward_if_max_step_reached if step_count == self.max_n_steps else fitness
