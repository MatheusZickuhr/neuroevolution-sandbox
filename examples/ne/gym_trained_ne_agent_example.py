
from python_ne.core.model_adapters.default_model_adapter import DefaultModelAdapter

from neuroevolution_sandbox.agents.ne_agent import NeAgent
from neuroevolution_sandbox.env_adapters.gym_env_adapter import GymEnvAdapter

if __name__ == '__main__':

    env_adapter = GymEnvAdapter(env_name='LunarLander-v2', render=False)

    agent = NeAgent(
        env_adapter=env_adapter,
        model_adapter=DefaultModelAdapter,
    )

    agent.load('ne_agent.json')
    agent.play()
