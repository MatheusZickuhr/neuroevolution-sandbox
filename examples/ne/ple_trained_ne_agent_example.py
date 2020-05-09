from python_ne.core.model_adapters.keras_model_adapter import KerasModelAdapter

from neuroevolution_sandbox.agents.ne_agent import NeAgent
from neuroevolution_sandbox.env_adapters.ple_env_adapter import PleEnvAdapter

if __name__ == '__main__':

    env_adapter = PleEnvAdapter(env_name='flappybird', render=True)

    agent = NeAgent(
        env_adapter=env_adapter,
        model_adapter=KerasModelAdapter,
    )

    agent.load('ne_agent.json')
    agent.play()
