import random

from neuroevolution_sandbox.env_adapters.env_adapter import EnvAdapter


class PleEnvAdapter(EnvAdapter):
    """Pygame learning env adapter"""

    def get_input_shape(self):
        return (len(self.env.getGameState()),)

    def reset(self):
        self.env.reset_game()

    def step(self, action) -> (object, float, bool):
        observation = self.env.getGameState()
        observation = [val for key, val in observation.items()]
        reward = self.env.act(self.env.getActionSet()[action])
        done = self.env.game_over()
        return observation, reward, done

    def get_n_actions(self) -> int:
        return len(self.env.getActionSet())

    def get_random_action(self):
        return random.randint(0, len(self.env.getActionSet()) - 1)
