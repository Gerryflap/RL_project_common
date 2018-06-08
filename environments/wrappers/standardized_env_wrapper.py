"""
    A wrapper that converts the standardized environments to a format that the DeepQ and Deep SARSA(λ) implementations
    made by Gerben accept.
"""
import core


class StandardizedEnvWrapper(object):
    def __init__(self, env: core.DiscreteEnvironment, state_transformer):
        """
        Initializes the standardized env wrapper
        :param env: The environment
        :param state_transformer: A function that extracts a numerical state representation from the Observation object.
        """
        self.env = env
        self.state_transformer = state_transformer
        self.terminated = True
        self.render = False
        self.action_space = self.env.valid_actions()

    def reset(self):
        self.terminated = False
        observation = self.env.reset()
        return self.state_transformer(observation)

    def step(self, action: core.Action):
        self.env.render = self.render
        observation, reward = self.env.step(action)
        self.terminated = observation.terminal
        return self.state_transformer(observation), reward

    def set_rendering(self, rendering):
        self.env.render = rendering
        self.render = rendering

