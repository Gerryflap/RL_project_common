from multiprocessing import Process

NUM_RUNS = 5  # Number of runs of each experiment over which will be averaged later


def snake_conv_sarsa(episodes=5000, file_name='snek'):
    import keras as ks
    import numpy as np
    from experiment_util import Logger
    from agents.deep_sarsa import DeepSarsa
    from environments.snake import SnakeVisual
    from q_network_sarsa_lambda import QNetworkSL

    logger = Logger(filename=file_name)

    env = SnakeVisual(render=False, render_freq=10)
    actions = env.valid_actions()
    size = np.shape(env.reset().state)

    nn = ks.models.Sequential()
    nn.add(ks.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid', input_shape=size))
    nn.add(ks.layers.Conv2D(filters=24, kernel_size=(5, 5), activation='sigmoid'))
    nn.add(ks.layers.Conv2D(filters=32, kernel_size=(5, 5), activation='sigmoid'))
    nn.add(ks.layers.Flatten())
    nn.add(ks.layers.Dense(units=16, activation='sigmoid'))
    nn.add(ks.layers.Dense(units=3, activation='linear'))

    nn.compile(optimizer=ks.optimizers.Adam(lr=0.0001), loss='mse')

    def normalize_state(s):
        return np.reshape(s.state, newshape=(1,) + size)

    dqn = QNetworkSL(nn, actions, normalize_state,
                     lambd=0.9,
                     gamma=0.9,
                     reward_factor=0.01,
                     fixed_length=100
                     )

    dql = DeepSarsa(env, dqn,
                    epsilon=0.3,
                    epsilon_step_factor=0.999,
                    epsilon_min=0.02,
                    replay_memory_size=1000
                    )

    experiment = logger.start_experiment(dql.get_configuration())
    q = dql.learn(num_episodes=episodes, result_handler=experiment.log)
    experiment.save_attribute("weights", nn.get_weights())


if __name__ == '__main__':
    jobs = [Process(target=snake_conv_sarsa,
                    args=(10, '../results/snake_continuous_conv_sarsa_run_' + str(i) + '.h5')) for i in range(NUM_RUNS)]
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()
