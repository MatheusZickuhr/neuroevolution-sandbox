import neat

from neuroevolution_sandbox.agents.neat_agent import NeatAgent
from neuroevolution_sandbox.env_adapters.ple_env_adapter import PleEnvAdapter

env_adapter = PleEnvAdapter(env_name='flappybird', render=False, continuous=False)
agent = NeatAgent(env_adapter=env_adapter, config_file_path='config.txt')

agent.train(
    number_of_generations=300,
    reporters=(
        neat.StdOutReporter(True),
        neat.StatisticsReporter(),
    )
)

agent.save('model')
