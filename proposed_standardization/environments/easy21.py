import random

from proposed_standardization.core import DiscreteEnvironment, Observation, Action

'''
    Easy21 environment implementation as defined in 
    http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf
    
'''


class Easy21Observation(Observation):
    """
        Easy21 Environment Observation
    """

    def __init__(self, p_sum: int, d_sum: int, terminal: bool):
        """
        Create a new Easy21 observation
        :param p_sum: The player score
        :param d_sum: The dealer score
        :param terminal: A boolean indicating whether the game is over
        """
        super().__init__(terminal)
        self.p_sum, self.d_sum = p_sum, d_sum

    def __str__(self):
        """
        :return: A string representation of this state
        """
        return 'Easy21(P: {:<3}, D: {:<3}, T: {})'.format(self.p_sum, self.d_sum, 'y' if self.terminal else 'n')

    def __eq__(self, other):
        """
        Defines equality between two states
        :param other: Object to compare this state with
        :return: Whether the specified object is equal to this state
        """
        if not isinstance(other, Easy21Observation):
            return False
        else:
            return self.p_sum == other.p_sum and \
                   self.d_sum == other.d_sum and \
                   self.terminal == other.terminal

    def __hash__(self) -> int:
        """
        :return: A unique hash corresponding to this state
        """
        h = 2 if self.terminal else 0
        h += self.p_sum * 3
        h += self.d_sum * 5
        return h


class Easy21Action(Action):
    """
        Easy21 Action that can be performed on the environment state
    """

    def __init__(self, hit: bool):
        """
        Create a new Easy21 action
        :param hit: A boolean indicating whether the player hits
        """
        self.hit = hit

    def __str__(self):
        return 'hit' if self.hit else 'stick'

    def __repr__(self):
        return str(self)


class Easy21(DiscreteEnvironment):
    """
        Easy21 Environment class
    """

    def __init__(self, p_red: float = 1 / 3):
        """
        Create a new Easy21 environment
        :param p_red: The probability of drawing a red card
        """
        super().__init__(Easy21Observation, Easy21Action)
        assert 0 <= p_red <= 1
        self.p_red = p_red

        hit = Easy21Action(True)
        stick = Easy21Action(False)
        self._actions = [stick, hit]

        self.p_sum, self.d_sum, self.terminal = self._draw_init_state()

    @property
    def state(self) -> tuple:
        """
        The environment state consists of the player score, dealer score and whether the state is terminal or not
        :return: a three-tuple containing
                    - Player score
                    - Dealer score
                    - Indication if the state is terminal
        """
        return self.p_sum, self.d_sum, self.terminal

    @staticmethod
    def _draw_card_value() -> int:
        """
        :return: A random card value
        """
        return random.randint(1, 10)

    @staticmethod
    def _card_value(card: tuple) -> int:
        """
        Get the effect of the specified card. Red cards subtract value, black cards add value
        :param card: The card for which its value should be calculated
        :return: The value of the specified card
        """
        v, c = card
        return v if c else -v

    @staticmethod
    def _valid_score(score: int) -> bool:
        """
        Checks whether the given score is greater or equal to 1 and smaller or equal than 21
        :param score: The score to be checked
        :return: Whether the score is valid
        """
        return 1 <= score <= 21

    def _draw_card_color(self) -> bool:
        """
        :return: A boolean specifying whether the color is black (with p(True)=p_red)
        """
        return random.random() > self.p_red

    def _draw_card(self, force_black: bool = False) -> tuple:
        """
        Draw a random card from the pile
        :param force_black: If set to true, the card drawn will always be black
        :return: The drawn card
        """
        if force_black:
            return self._draw_card_value(), True
        return self._draw_card_value(), self._draw_card_color()

    def _draw_init_state(self) -> tuple:
        """
        :return: A random initial state for Easy21
        """
        return self._card_value(self._draw_card(force_black=True)),\
               self._card_value(self._draw_card(force_black=True)),\
               False

    def _reward(self, p_sum, d_sum, terminal) -> int:
        """
        Determine the reward of the given state
        :param p_sum: Player score of the state
        :param d_sum: Dealer score of the state
        :param terminal: Indication if the state is terminal
        :return: a reward for the state
        """
        if not terminal:                                    # Only terminal states give a nonzero reward
            return 0
        if not self._valid_score(p_sum):                    # Invalid player score -> -1 reward
            return -1
        if not self._valid_score(d_sum):                    # Invalid dealer score -> +1 reward
            return 1
        if p_sum > d_sum:                                   # Player score > Dealer score -> +1 reward
            return 1
        if p_sum < d_sum:                                   # Player score < Dealer score -> -1 reward
            return -1
        return 0                                            # Draw -> 0 reward

    def step(self, action: Easy21Action) -> tuple:
        """
        Perform an action on the environment state
        :param action: The action to be performed
        :return: A two-tuple of (observation, reward)
        """
        if self.terminal:
            raise Exception('Cannot perform action on terminal state!')
        if action.hit:                                                          # Case 1: Hit action
            self.p_sum += self._card_value(self._draw_card())                   # - Draw a card for the player
            self.terminal = not self._valid_score(self.p_sum)                   # - Check if score is still valid
        else:                                                                   # Case 2: Stick action -> play dealer
            while self._valid_score(self.d_sum) and self.d_sum < 17:            # - While dealer sum is below 17
                self.d_sum += self._card_value(self._draw_card())               # - Keep drawing cards
            self.terminal = True                                                # - End game
        return Easy21Observation(*self.state), self._reward(*self.state)        # Return observation and reward

    def reset(self) -> Easy21Observation:
        """
        Reset the environment state
        :return: an initial observation
        """
        self.p_sum, self.d_sum, self.terminal = self._draw_init_state()
        return Easy21Observation(*self.state)

    def valid_actions(self) -> list:
        """
        :return: A list of actions that can be performed on the current environment state
        """
        return list(self._actions)


if __name__ == '__main__':
    from collections import Counter

    _env = Easy21()                                 # Create an Easy21 environment
    _actions = _env.valid_actions()

    _a = _env.sample()                              # Sample a random action from the environment
    _o, _r = _env.step(_a)                          # Perform the action

    print(_actions)
    print(_a)
    print(_o, _r)

    _rs = []                                        # Play a number of random games. Keep track of rewards
    for _ in range(10000):
        _o = _env.reset()
        while not _o.is_terminal():
            _a = _env.sample()
            _o, _r = _env.step(_a)
        _rs += [_r]
    print(Counter(_rs))                             # Show outcome distribution
