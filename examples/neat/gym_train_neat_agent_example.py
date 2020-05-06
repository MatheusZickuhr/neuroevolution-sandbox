import gym
import neat

from neuroevolution_sandbox.agents.neat_agent import NeatAgent
from neuroevolution_sandbox.env_adapters.gym_env_adapter import GymEnvAdapter

env = gym.make('LunarLander-v2')

env_adapter = GymEnvAdapter(env=env, render=False, continuous=False)
agent = NeatAgent(env_adapter=env_adapter, config_file_path='config.txt')

agent.train(
    number_of_generations=300,
    play_n_times=5,
    max_n_steps=300,
    reward_if_max_step_reached=-200,
    reporters=(
        neat.StdOutReporter(True),
        neat.StatisticsReporter(),
        neat.Checkpointer(5)
    )
)
