import os
import random

from ple import PLE

from neuroevolution_sandbox.env_adapters.env_adapter import EnvAdapter
from ple.games.pong import Pong
from ple.games.catcher import Catcher
from ple.games.pixelcopter import Pixelcopter
from ple.games.flappybird import FlappyBird
from ple.games.monsterkong import MonsterKong
from ple.games.puckworld import PuckWorld
from ple.games.raycastmaze import RaycastMaze
from ple.games.snake import Snake
from ple.games.waterworld import WaterWorld

envs_lookup_table = {
    'pong': Pong,
    'catcher': Catcher,
    'pixelcopter': Pixelcopter,
    'flappybird': FlappyBird,
    'monsterkong': MonsterKong,
    'puckworld': PuckWorld,
    'raycastmaze': RaycastMaze,
    'snake': Snake,
    'waterworld': WaterWorld,
}


class PleEnvAdapter(EnvAdapter):
    """Pygame learning env adapter"""

    def __init__(self, *args, **kwargs):
        super(PleEnvAdapter, self).__init__(*args, **kwargs)

        if not self.render:
            os.putenv('SDL_VIDEODRIVER', 'fbcon')
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        Game = envs_lookup_table[self.env_name]
        self.env = PLE(Game(), display_screen=self.render, force_fps=not self.render)
        self.env.init()

    def get_input_shape(self):
        return (len(self.env.getGameState()),)

    def reset(self):
        self.env.reset_game()

    def step(self, action) -> (object, float, bool):
        reward = self.env.act(self.env.getActionSet()[action])
        observation = self.env.getGameState()
        observation = [val for key, val in observation.items()]
        done = self.env.game_over()
        return observation, reward, done

    def get_n_actions(self) -> int:
        return len(self.env.getActionSet())

    def get_random_action(self):
        return random.randint(0, len(self.env.getActionSet()) - 1)
