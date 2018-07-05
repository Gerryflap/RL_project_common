
if __name__ == '__main__':
    import keras as ks
    import numpy as np
    from agents.deep_q import DeepQLearning
    from environments.cartpole import CartPole
    from q_network import QNetwork

    nn = ks.models.Sequential()
    nn.add(ks.layers.Dense(32, activation='sigmoid', input_shape=(4,)))
    nn.add(ks.layers.Dense(32, activation='sigmoid'))
    nn.add(ks.layers.Dense(2, activation='linear'))

    nn.compile(optimizer=ks.optimizers.Adam(lr=0.001),
               loss='mse')

    env = CartPole(render=True)
    actions = env.valid_actions()

    dqn = QNetwork(nn, actions, lambda x: np.reshape(x.state, newshape=(1, 4)))

    dql = DeepQLearning(env, dqn)

    q = dql.learn()
