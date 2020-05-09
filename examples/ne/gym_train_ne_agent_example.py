
from python_ne.core.ga.console_logger import ConsoleLogger
from python_ne.core.ga.crossover_strategies import NoCrossover
from python_ne.core.ga.mutation_strategies import Mutation1
from python_ne.core.model_adapters.default_model_adapter import DefaultModelAdapter

from neuroevolution_sandbox.agents.ne_agent import NeAgent
from neuroevolution_sandbox.env_adapters.gym_env_adapter import GymEnvAdapter

if __name__ == '__main__':

    env_adapter = GymEnvAdapter(env_name='LunarLander-v2', render=True, continuous=False)

    agent = NeAgent(
        env_adapter=env_adapter,
        model_adapter=DefaultModelAdapter,
    )

    nn_config = (
        (env_adapter.get_input_shape(), 16, 'tanh'),
        (env_adapter.get_n_actions(), 'tanh')
    )

    agent.train(
        number_of_generations=1000,
        population_size=500,
        selection_percentage=0.9,
        mutation_chance=0.01,
        fitness_threshold=250,
        neural_network_config=nn_config,
        crossover_strategy=NoCrossover(),
        mutation_strategy=Mutation1(),
        loggers=(ConsoleLogger(),),
        play_n_times=5,
        max_n_steps=300,
        reward_if_max_step_reached=-200
    )

    # agent.save('ne_agent.json')
