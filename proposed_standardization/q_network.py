import numpy as np
import keras as ks

from proposed_standardization.q_estimator import QEstimator


class QNetwork(QEstimator):
    """
        Deep Q-Network class that wraps around a suitable Keras model
    """

    def __init__(self, model: ks.Model, out_map: list, feature_ex: callable=lambda x: x, gamma: float=1.0):
        """
        Create a new Deep Q-Network
        :param model: Keras model to be used in the network
        :param out_map: A list of actions mapping the model's output to actions (by index)
        :param feature_ex: Function that is applied on a state to transform it into suitable input for the model
        :param gamma: Reward discount factor
        """
        assert (None, len(out_map)) == model.output_shape  # Make sure all outputs can be mapped to actions
        self.model = model
        self.phi = feature_ex
        self.out_map = out_map
        self.gamma = gamma

    def Q(self, observation, action) -> float:
        """
        Predict a Q value for the observation action pair
        :param observation: The observation parameter
        :param action: The action parameter
        :return: Q(s, a)
        """
        out = self.model.predict([self.phi(observation)])[0]    # Predict Q for the transformed state and all actions
        return out[self.out_map.index(action)]                  # Return the relevant Q value

    def Qs(self, observation, actions=None) -> dict:
        """
        Predict the Q values for all actions that can be performed from a state/observation
        :param observation: The observation parameter
        :param actions: Optional list of allowed actions that the result should be restricted to.
                        If left unspecified, all actions in the action space are given
        :return: A dictionary mapping all actions to their expected Q value
        """
        pi = dict()
        for i, v in enumerate(self.model.predict([self.phi(observation)])[0]):  # Perform a forward pass in the network
            pi[self.out_map[i]] = v                                             # Map actions to their predicted Q-value
        if actions is None:
            return pi
        else:
            return {a: v for a, v in pi.items() if a in actions}

    def fit_on_samples(self, samples):
        """
        Train the network on a minibatch of samples
        :param samples: A list of four-tuples (State, Action, Reward, Next State)
        """
        for s, a, r, s_p in samples:                                             # Iterate through all samples
            phi_s = self.phi(s)                                                  # Prepare state for model input
            phi_p = self.phi(s_p)
            qs = self.model.predict([phi_s])[0]                                  # Compute model output
            qp = self.model.predict([phi_p])[0]
            if s_p.is_terminal():                                                # Get model target
                qs[self.out_map.index(a)] = r
            else:
                qs[self.out_map.index(a)] = r + self.gamma * max(qp)
            self.model.fit(x=phi_s,                                              # Train model on target
                           y=np.reshape(qs, newshape=(1, len(self.out_map))),
                           epochs=1,
                           verbose=2
                           )


if __name__ == '__main__':
    import numpy as np

    _model = ks.models.Sequential()
    _model.add(ks.layers.Dense(64, activation='sigmoid', input_shape=(16,)))
    _model.add(ks.layers.Dense(2, activation='sigmoid'))

    _model.compile(optimizer='sgd', loss='mse')

    _out_map = [False, True]

    _feature_ex = lambda x: x / 10

    dqn = QNetwork(_model, _out_map, _feature_ex)

    _s = np.random.random(size=(1, 16))

    print(dqn.Qs(_s, [False, True]))
    print(dqn.Q(_s, False))
