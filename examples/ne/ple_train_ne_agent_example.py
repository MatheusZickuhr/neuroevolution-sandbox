from python_ne.core.ga.console_logger import ConsoleLogger
from python_ne.core.ga.crossover_strategies import Crossover4
from python_ne.core.ga.mutation_strategies import Mutation1
from python_ne.core.model_adapters.default_model_adapter import DefaultModelAdapter

from neuroevolution_sandbox.agents.ne_agent import NeAgent
from neuroevolution_sandbox.env_adapters.ple_env_adapter import PleEnvAdapter

if __name__ == '__main__':
    env_adapter = PleEnvAdapter(env_name='flappybird', render=True)

    agent = NeAgent(
        env_adapter=env_adapter,
        model_adapter=DefaultModelAdapter,
    )

    agent.train(
        number_of_generations=150,
        population_size=500,
        selection_percentage=0.9,
        mutation_chance=0.01,
        fitness_threshold=50,
        crossover_strategy=Crossover4(),
        mutation_strategy=Mutation1(),
        neural_network_config=[
            (env_adapter.get_input_shape(), 16, 'tanh'),
            (env_adapter.get_n_actions(), 'tanh')
        ],
        loggers=(ConsoleLogger(),)
    )

    agent.save('ne_agent.json')
