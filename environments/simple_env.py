from core import FiniteActionEnvironment, Action, State


class SimpleEnvAction(Action):
    def __init__(self, direction: int):
        self.dir = direction

    def __repr__(self):
        return "Action(%d)" % self.dir


class SimpleEnvState(State):
    def __init__(self, terminal: bool, n: int):
        super().__init__(terminal)
        self.n = n


A_LEFT = SimpleEnvAction(-1)
A_NOOP = SimpleEnvAction(0)
A_RIGHT = SimpleEnvAction(1)

ACTION_SPACE = [A_LEFT, A_NOOP, A_RIGHT]


class SimpleEnv(FiniteActionEnvironment):
    def __init__(self):
        self.state = 0
        self.term = True
        self.steps = 0

    @staticmethod
    def valid_actions_from(state) -> list:
        return ACTION_SPACE

    @staticmethod
    def action_space() -> list:
        return ACTION_SPACE

    def valid_actions(self) -> list:
        return ACTION_SPACE

    def step(self, action: SimpleEnvAction) -> tuple:
        if self.term:
            raise Exception("Trying to step on Terminal state")
        self.state += action.dir
        self.steps += 1

        term = self.state > 1 or self.state < -1 or self.steps > 50
        if term:
            print(self.steps)

        self.term = term
        return SimpleEnvState(term, self.state), 1

    def reset(self):
        self.term = False
        self.state = 0
        self.steps = 0
        return SimpleEnvState(False, self.state)

