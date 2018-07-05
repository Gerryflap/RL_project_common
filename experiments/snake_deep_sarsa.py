
if __name__ == '__main__':
    import keras as ks
    import numpy as np
    from agents.deep_sarsa import DeepSarsa
    from environments.snake import SnakeContinuous
    from q_network_sarsa_lambda import QNetworkSL

    neural_network = ks.models.Sequential()
    neural_network.add(ks.layers.Dense(150, activation='relu', input_shape=(9,)))
    neural_network.add(ks.layers.Dense(50, activation='relu'))
    neural_network.add(ks.layers.Dense(3, activation='linear'))

    neural_network.compile(optimizer=ks.optimizers.Adam(lr=0.001), loss='mse')

    env = SnakeContinuous(grid_size=[8, 8], render=True, render_freq=10)
    actions = env.valid_actions()

    dqn = QNetworkSL(neural_network, actions, lambda x: np.reshape(x.state, newshape=(1, 9)),
                     lambd=0.9,
                     gamma=0.9,
                     reward_factor=0.01,
                     fixed_length=100
                     )

    dql = DeepSarsa(env, dqn,
                    epsilon=0.3,
                    epsilon_step_factor=0.999,
                    epsilon_min=0.05,
                    replay_memory_size=1000
                    )

    q = dql.learn()
