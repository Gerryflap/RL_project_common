from multiprocessing import Pool, TimeoutError
import numpy as np
import time
import os
"""
    Runs the Cartpole Deep SARSA lambda experiments
"""

def experiment(run_n, episodes, sigmas, lambda_parameter):
    """
    Runs a single experiment for each sigma value of Deep SARSA lambda on Cartpole
    :param run_n: The run number, used in the filename of the experiment
    :param episodes: Number of epsiodes to run
    :param sigmas: Values of sigma (noise standard deviation)
    :param lambda_parameter: The lambda value for this experiment
    :return: The filename of the output file
    """
    import tensorflow as tf

    # This code is used to stop tensorflow from allocating all GPU memory ar once. This allows for more runs on one GPU
    # These settings are ignored when running on CPU (which is often faster for this experiment)
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    set_session(tf.Session(config=config))
    import keras as ks


    import numpy as np
    from agents.deep_sarsa import DeepSarsa
    from environments.cartpole import NoisyCartPole
    from q_network_sarsa_lambda import QNetworkSL

    from experiment_util import Logger

    filename = ("results/cartpole_deepsarsalambda_lambda_%1.2f_%d.h5" %(lambda_parameter, run_n))
    l = Logger(filename= filename)


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
    # Experiment parameters:
    runs = 5
    episodes = 250
    sigmas = np.array([0, 10**-2, 10**-1, 10**-0])
    lambdas = np.array([0, 0.5, 0.75, 0.9, 1])

    # If more CPUs are available, change the 4 here to the number of desired threads:
    with Pool(processes=4) as pool:
        for i in pool.starmap(experiment, [(run_n, episodes, sigmas, l) for l in lambdas for run_n in range(runs)]):
            print("Finished %s" % i) 
    
    
