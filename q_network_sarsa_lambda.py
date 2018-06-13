import numpy as np
import keras as ks

from q_estimator import QEstimator


class QNetworkSL(QEstimator):
    """
        Deep SARSA(λ)-Network class that wraps around a suitable Keras model
    """

    def __init__(self, model: ks.Model, out_map: list, feature_ex: callable=lambda x: x, gamma: float=1.0, lambd: float=0.0, fixed_length=100):
        """
        Create a new Deep SARSA(λ)-Network
        :param model: Keras model to be used in the network
        :param out_map: A list of actions mapping the model's output to actions (by index)
        :param feature_ex: Function that is applied on a state to transform it into suitable input for the model
        :param gamma: Reward discount factor
        """
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

    def fit_on_trajectories(self, trajectories):
        """
        Train the network on a minibatch of trajectories
        :param trajectories: A list of Trajectories consisting of three-tuples (State, Action, Reward)
        """
        initial_states = list([t[0][0] for t in trajectories])

        # Initialize the target Q-values with the current output
        target_qs = self.live_model.predict(initial_states)

        for i, trajectory in enumerate(trajectories):
            states = list([t[0] for t in trajectory])
            if len(states) > 1:
                # Shorten the states such that the current state is not in there
                fixed_qs = self.live_model.predict(states[1:])

            q_return = 0
            total_discounted_reward = 0

            for j in range(len(trajectory)):
                s, a, r = trajectory[j]

                # Convert back to action index:
                a = self.action_index_map[a]

                terminal_state = j == len(trajectory) - 1
                if not terminal_state:
                    sp, ap, _ = trajectory[j+1]

                    # Convert back to action index:
                    ap = self.action_index_map[ap]

                total_discounted_reward += r * self.gamma ** j

                # Calculate the TD(n) return for n = j
                if terminal_state:
                    n_return = total_discounted_reward
                else:
                    n_return = total_discounted_reward + self.gamma**(j+1) * fixed_qs[j, ap]
                q_return += self.lambd ** j * n_return

            action = trajectory[0][1]
            action = self.action_index_map[action]

            target_qs[i, action] = q_return
        self.live_model.fit(initial_states, target_qs, verbose=False)

        self.training_steps_since_sync += 1

        # If we have reached a certain number of updates, sync the fixed weights
        if self.training_steps_since_sync >= self.fixed_length:
            self.fixed_model.set_weights(self.live_model.get_weights())









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
