from neuroevolution_sandbox.env_adapters.env_adapter import EnvAdapter


class GymEnvAdapter(EnvAdapter):

    def get_continuous_space_len(self):
        return self.env.action_space.shape[0]

    def get_input_shape(self):
        return self.env.observation_space.shape

    def reset(self):
        self.env.reset()

    def step(self, action):
        if self.render:
            self.env.render()

        observation, reward, done, info = self.env.step(action)
        return observation, float(reward), done

    def get_n_actions(self):
        return self.env.action_space.n

    def get_random_action(self):
        return self.env.action_space.sample()
