import math

import numpy as np
import keras as ks

from q_estimator import QEstimator


class QNetworkSL(QEstimator):
    """
        Deep SARSA(λ)-Network class that wraps around a suitable Keras model
    """

    def __init__(
            self,
            model: ks.Model,
            out_map: list,
            feature_ex: callable=lambda x: x,
            gamma: float=1.0,
            lambd: float=0.0,
            fixed_length=100,
            reward_factor=1.0,
            fix_lambda_skew=True,
            lambda_min=0.0
    ):
        """
        Create a new Deep SARSA(λ)-Network
        :param model: Keras model to be used in the network
        :param out_map: A list of actions mapping the model's output to actions (by index)
        :param feature_ex: Function that is applied on a state to transform it into suitable input for the model
        :param gamma: Reward discount factor
        :param lambd: Lambda value
        :param fixed_length: Denotes the number of steps before the fixed network gets synced with the live network
        :param reward_factor: A scaling factor on the rewards. Allows for scaling the rewards if they become too large
        :param fix_lambda_skew: Fixes an issue arising from finite trajectories.
            Officially the lambda returns are scaled with (1-λ) * λ^k, which would result in a sum of 1 when taking
            k to infinity. In a practical situation, trajectories are not infinite an thus this scaling needs to be
            different. When this parameter is set to true, the total return will be divided by the sum of
            lambdas (λ^0 + λ^1 ... λ^n) instead. This will make the sum of lambda weights 1 again.
        :param lambda_min: Lambda_min defines a minimum lambda value after which calculations are considered irrelevant
            When calculating the new Q-values, the TD(λ) returns will be calculated up until the point where λ^k < λ_min.
            This massively speeds up computations in cases where the trajectories become long (eg. >100) but loses
            a bit of precision.
        """
        self.lambda_min = lambda_min
        self.max_trajectory_length = None
        if self.lambda_min != 0:
            self.max_trajectory_length = int(math.log(self.lambda_min, lambd))
            print("Capping trajectories at a max length of ", self.max_trajectory_length)
        self.fix_lambda_skew = fix_lambda_skew
        assert (None, len(out_map)) == model.output_shape  # Make sure all outputs can be mapped to actions
        self.live_model = model
        # Copy the model config:
        self.fixed_model = ks.models.model_from_json(model.to_json())
        self.phi = feature_ex
        self.out_map = out_map
        self.action_index_map = dict()
        for i, a in enumerate(out_map):
            self.action_index_map[a] = i
        self.gamma = gamma
        self.lambd = lambd
        self.fixed_length = fixed_length
        self.training_steps_since_sync = 0
        self.reward_factor = reward_factor

    def Q(self, state, action) -> float:
        """
        Predict a Q value for the state action pair
        :param state: The state parameter
        :param action: The action parameter
        :return: Q(s, a)
        """
        out = self.live_model.predict([self.phi(state)])[0]    # Predict Q for the transformed state and all actions
        return out[self.out_map.index(action)]                  # Return the relevant Q value

    def Qs(self, state, actions=None) -> dict:
        """
        Predict the Q values for all actions that can be performed from a state
        :param state: The state parameter
        :param actions: Optional list of allowed actions that the result should be restricted to.
                        If left unspecified, all actions in the action space are given
        :return: A dictionary mapping all actions to their expected Q value
        """
        pi = dict()
        for i, v in enumerate(self.live_model.predict([self.phi(state)])[0]):  # Perform a forward pass in the network
            pi[self.out_map[i]] = v                                             # Map actions to their predicted Q-value
        if actions is None:
            return pi
        else:
            return {a: v for a, v in pi.items() if a in actions}

    def Q_array(self, states):
        return self.live_model.predict(states)

    def Qp_array(self, states):
        return self.fixed_model.predict(states)

    def fit_on_trajectories(self, trajectories):
        """
        Train the network on a minibatch of trajectories
        :param trajectories: A list of Trajectories consisting of three-tuples (State, Action, Reward)
        """
        initial_states = np.array([self.phi(t[0][0])[0] for t in trajectories])

        # Initialize the target Q-values with the current output
        target_qs = self.live_model.predict(initial_states)

        # Calculate the target q values for each of the trajectories in the minibatch:
        for i, trajectory in enumerate(trajectories):

            # Get all states in the trajectory (or up to a certain point when using lambda_min):
            if self.max_trajectory_length is None:
                states = np.array([self.phi(t[0])[0] for t in trajectory])
            else:
                states = np.array([self.phi(t[0])[0] for t in trajectory[:self.max_trajectory_length+1]])

            if len(states) > 1:
                # Shorten the states such that the current state is not in there
                fixed_qs = self.live_model.predict(states[1:])

            q_return = 0
            total_discounted_reward = 0
            # Used for fixing the "lambda skew" caused by the infinity assumption in the lambda return function
            lambda_sum = 0

            # Calculate all TD(n) returns and add them to the total to get TD(lambda)

            iterations = len(trajectory)
            if self.max_trajectory_length is not None:
                iterations = min(iterations, self.max_trajectory_length)

            for j in range(iterations):
                s, a, r = trajectory[j]

                r *= self.reward_factor

                terminal_state = j == len(trajectory) - 1
                if not terminal_state:
                    _, ap, _ = trajectory[j+1]

                    # Convert back to action index:
                    ap = self.action_index_map[ap]

                total_discounted_reward += r * self.gamma ** j

                # Calculate the TD(n) return for n = j
                if terminal_state:
                    n_return = total_discounted_reward
                else:
                    n_return = total_discounted_reward + self.gamma**(j+1) * fixed_qs[j, ap]
                q_return += self.lambd ** j * n_return
                lambda_sum += self.lambd ** j

            if not self.fix_lambda_skew:
                q_return *= (1 - self.lambd)
            else:
                q_return /= lambda_sum

            action = trajectory[0][1]
            action = self.action_index_map[action]

            target_qs[i, action] = q_return
        self.live_model.fit(initial_states, target_qs, verbose=False)

        self.training_steps_since_sync += 1

        # If we have reached a certain number of updates, sync the fixed weights
        if self.training_steps_since_sync >= self.fixed_length:
            self.fixed_model.set_weights(self.live_model.get_weights())
            self.training_steps_since_sync = 0










if __name__ == '__main__':
    import numpy as np

    _model = ks.models.Sequential()
    _model.add(ks.layers.Dense(64, activation='sigmoid', input_shape=(16,)))
    _model.add(ks.layers.Dense(2, activation='sigmoid'))

    _model.compile(optimizer='sgd', loss='mse')

    _out_map = [False, True]

    _feature_ex = lambda x: x / 10

    dqn = QNetworkSL(_model, _out_map, _feature_ex)

    _s = np.random.random(size=(1, 16))

    print(dqn.Qs(_s, [False, True]))
    print(dqn.Q(_s, False))
