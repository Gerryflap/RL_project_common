from multiprocessing import Pool, TimeoutError
import numpy as np
import time
import os


def experiment(runs, episodes, sigmas, lambda_parameter):
    import keras as ks
    import numpy as np
    from agents.deep_sarsa import DeepSarsa
    from environments.cartpole import NoisyCartPole
    from q_network_sarsa_lambda import QNetworkSL

    from experiment_util import Logger

    filename = ("results/cartpole_deepsarsalambda_lambda_%1.2f.h5" %lambda_parameter)
    l = Logger(filename= filename)

    for run_n in range(runs):
        for sigma in sigmas:
            
            neural_network = ks.models.Sequential()
            neural_network.add(ks.layers.Dense(150, activation='relu', input_shape=(4,)))
            neural_network.add(ks.layers.Dense(50, activation='relu'))
            neural_network.add(ks.layers.Dense(2, activation='linear'))

            neural_network.compile(optimizer=ks.optimizers.Adam(lr=0.001),
                                   loss='mse')
                    
            env = NoisyCartPole(std= sigma, render=False)
            actions = env.valid_actions()
            
            dqn = QNetworkSL(neural_network, actions, lambda x: np.reshape(x.state, newshape=(1, 4)),
                             lambd=lambda_parameter,
                             lambda_min=1e-3,
                             gamma=1.0,
                             reward_factor=0.01,
                             fixed_length=100)

            dql = DeepSarsa(env, dqn,
                            epsilon=1.0,
                            epsilon_step_factor=0.9995,
                            epsilon_min=0.0,
                            replay_memory_size=1000
            )

            c = dql.get_configuration()
            experiment = l.start_experiment( c )
            q = dql.learn( num_episodes=episodes, result_handler=experiment.log)
            experiment.save_attribute("weights", neural_network.get_weights())
            print("%s finished sigma=%1.2f, run=%i" % (filename, sigma, run_n) )
    return filename

if __name__ == "__main__":
    runs = 5
    episodes = 500
    sigmas = np.array([0, 10**-2, 10**-1, 10**-0])
    lambdas = np.array([0, 0.5, 0.75, 0.9, 1])

    with Pool(processes=32) as pool:
        for i in pool.starmap(experiment, [(runs, episodes, sigmas, l) for l in lambdas]):
            print("Finished %s" % i) 
    
    
