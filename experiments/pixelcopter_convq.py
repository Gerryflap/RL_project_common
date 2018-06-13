if __name__ == '__main__':
    import keras as ks
    import numpy as np

    from version4.agents.deep_q import DeepQLearning
    from version4.environments.pixelcopter import VisualPixelCopter
    from version4.q_network import QNetwork

    width, height = size = (32, 32)
    env = VisualPixelCopter(size)
    actions = env.valid_actions()

    nn = ks.models.Sequential()
    nn.add(ks.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid', input_shape=size + (1,)))
    nn.add(ks.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid'))
    nn.add(ks.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid'))
    nn.add(ks.layers.Flatten())
    nn.add(ks.layers.Dense(units=16, activation='sigmoid'))
    nn.add(ks.layers.Dense(units=2,  activation='linear'))

    nn.compile(optimizer=ks.optimizers.Adam(lr=0.0001),
               loss='mse')

    print(nn.summary())

    def normalize_state(s):
        return np.reshape(s.observation / 256, newshape=(1,) + size + (1,))


    dqn = QNetwork(nn, actions, normalize_state)

    dql = DeepQLearning(env, dqn)

    q = dql.learn()
