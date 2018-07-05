import gym

from core import State, Action, FiniteActionEnvironment
from sacx.extcore import TaskEnvironment, Task

"""
    Environment wrapper for OpenAI Gym's MountainCar
"""

gym.envs.register(
    id='MountainCarLong-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    reward_threshold=10000.00,
)


class MountainCarState(State):
    """
        MountainCarState
    """

    def __init__(self, state, terminal: bool):
        """
        Create a new MountainCar State
        :param state: An state obtained from the OpenAI environment
        :param terminal: A boolean indicating whether the environment state is terminal
        """
        super().__init__(terminal)
        self.state = state

    def __str__(self) -> str:
        return str(self.state)


class MountainCarAction(Action):
    """
        MountainCar Environment Action
    """

    def __init__(self, value):
        """
        Create a new MountainCar Action
        :param direction: A boolean indicating the direction of the action (left=False, right=True)
        """
        self.value = value


MAIN_TASK = Task("main")
GO_RIGHT = Task("Go right")
GO_LEFT = Task("Go left")
GO_FAST = Task("Go fast")

TASKS =[MAIN_TASK, GO_RIGHT, GO_LEFT, GO_FAST]


class MountainCar(FiniteActionEnvironment, TaskEnvironment):
    """
        MountainCar environment class
    """

    LEFT = MountainCarAction(0)
    NO_OP = MountainCarAction(1)
    RIGHT = MountainCarAction(2)
    ACTIONS = [LEFT, NO_OP, RIGHT]



    def __init__(self, render=True):
        """
        Create a new MountainCarEnvironment
        :param render: A boolean indicating whether the environment should be rendered
        """
        super().__init__()
        self.env = gym.make('MountainCarLong-v0')
        self.render = render

        self.terminal = False
        self.step_v = 0

        self.reset()

    @staticmethod
    def action_space() -> list:
        return list(MountainCar.ACTIONS)

    @staticmethod
    def valid_actions_from(state) -> list:
        return MountainCar.action_space()

    def valid_actions(self) -> list:
        return MountainCar.action_space()

    def step(self, action: MountainCarAction) -> tuple:
        """
        Perform an action on the current environment state
        :param action: The action to be performed
        :return: A two-tuple of (state, reward)
        """
        if self.terminal:
            raise Exception('Cannot perform action on terminal state!')
        if self.render:
            self.env.render()
        state, reward, self.terminal, info = self.env.step(action.value)
        self.step_v += 1
        rewards = {
            MAIN_TASK: 1 if self.terminal and self.step_v < 1000 else 0,           # Make it a true sparse reward task
            GO_LEFT: 1 if action == self.LEFT else 0,
            GO_RIGHT: 1 if action == self.RIGHT else 0,
            GO_FAST: 1 if abs(state[1] >= 0.03) else 0
        }

        return MountainCarState(state, self.terminal), rewards

    def reset(self):
        """
        Reset the environment state
        :return: A state containing the initial state
        """
        self.terminal = False
        self.step_v = 0
        return MountainCarState(self.env.reset(), self.terminal)

    @staticmethod
    def auxiliary_tasks() -> list:
        return TASKS[1:]

    @staticmethod
    def get_tasks():
        return TASKS




if __name__ == '__main__':

    _e = MountainCar(render=True)
    _s = _e.reset()

    for _ in range(1000):
        while not _s.is_terminal():
            _s, _r = _e.step(_e.sample())
        _s = _e.reset()
