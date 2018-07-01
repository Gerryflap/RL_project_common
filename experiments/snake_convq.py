if __name__ == '__main__':
    import keras as ks
    import numpy as np

    from agents.deep_q import DeepQLearning
    from environments.snake import SnakeVisual
    from q_network import QNetwork

    env = SnakeVisual(grid_size=[8, 8], render=True, render_freq=1)
    actions = env.valid_actions()
    size = np.shape(env.reset().state)

    nn = ks.models.Sequential()
    nn.add(ks.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='sigmoid', input_shape=size))
    nn.add(ks.layers.Conv2D(filters=24, kernel_size=(3, 3), activation='sigmoid'))
    nn.add(ks.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='sigmoid'))
    nn.add(ks.layers.Flatten())
    nn.add(ks.layers.Dense(units=16, activation='sigmoid'))
    nn.add(ks.layers.Dense(units=3,  activation='linear'))

    nn.compile(optimizer=ks.optimizers.Adam(lr=0.0001), loss='mse')

    print(nn.summary())

    def normalize_state(s):
        return np.reshape(s.state, newshape=(1,) + size)


    dqn = QNetwork(nn, actions, normalize_state)

    dql = DeepQLearning(env, dqn)

    q = dql.learn()
