from python_ne.core.model_adapters.keras_model_adapter import KerasModelAdapter

from ple.games.flappybird import FlappyBird
from ple import PLE

from neuroevolution_sandbox.agents.ne_agent import NeAgent
from neuroevolution_sandbox.env_adapters.ple_env_adapter import PleEnvAdapter

if __name__ == '__main__':
    env = PLE(FlappyBird(), display_screen=True, force_fps=False)
    env.init()

    env_adapter = PleEnvAdapter(env=env)

    agent = NeAgent(
        env_adapter=env_adapter,
        model_adapter=KerasModelAdapter,
    )

    agent.load('ne_agent.json')
    agent.play()
