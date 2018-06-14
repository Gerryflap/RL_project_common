
if __name__ == '__main__':
    import keras as ks
    import numpy as np
    from agents.deep_sarsa import DeepSarsa
    from environments.cartpole import CartPole
    from q_network_sarsa_lambda import QNetworkSL

    neural_network = ks.models.Sequential()
    neural_network.add(ks.layers.Dense(150, activation='relu', input_shape=(4,)))
    neural_network.add(ks.layers.Dense(50, activation='relu'))
    neural_network.add(ks.layers.Dense(2, activation='linear'))

    neural_network.compile(optimizer=ks.optimizers.Adam(lr=0.001),
               loss='mse')

    env = CartPole(render=True)
    actions = env.valid_actions()

    dqn = QNetworkSL(neural_network, actions, lambda x: np.reshape(x.state, newshape=(1, 4)),
                     lambd=0.9,
                     gamma=0.9,
                     reward_factor=0.01,
                     fixed_length=100
                     )

    dql = DeepSarsa(env, dqn,
                    epsilon=0.3,
                    epsilon_step_factor=0.99995,
                    epsilon_min=0.05,
                    replay_memory_size=1000
                    )

    q = dql.learn()
