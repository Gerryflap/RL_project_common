import multiprocessing as mp


def experiment(runs, episodes, sigmas, lambd):
    import keras as ks
    import numpy as np
    from agents.deep_sarsa import DeepSarsa
    from environments.cartpole import CartPole
    from q_network_sarsa_lambda import QNetworkSL

    from experiment_util import Logger
    l = Logger()

    for run_n in range(runs):
        for sigma in sigmas:
            
            neural_network = ks.models.Sequential()
            neural_network.add(ks.layers.Dense(150, activation='relu', input_shape=(4,)))
            neural_network.add(ks.layers.Dense(50, activation='relu'))
            neural_network.add(ks.layers.Dense(2, activation='linear'))

            neural_network.compile(optimizer=ks.optimizers.Adam(lr=0.001),
                                   loss='mse')
                    
            env = CartPole(render=False)
            actions = env.valid_actions()
            
            dqn = QNetworkSL(neural_network, actions, lambda x: np.reshape(x.state, newshape=(1, 4)),
                             lambd=lambd[i],
                             gamma=0.9,
                             reward_factor=0.01,
                             fixed_length=100)

            dql = DeepSarsa(env, dqn,
                            epsilon=0.3,
                            epsilon_step_factor=0.99995,
                            epsilon_min=0.05,
                            replay_memory_size=1000
            )

            c = dql.get_configuration()
            print(c)
            experiment = l.start_experiment( c )
            q = dql.learn( num_episodes=200, result_handler=experiment.log)
            experiment.save_attribute("weights", neural_network.get_weights())

if __name__ == "__main__":
    runs = 5
    episodes = 500
    sigmas = np.array([0 10e-2 10e-1 10e-0])
    lambdas = np.array([0 0.5 0.75 0.9 1])

    
    
    
