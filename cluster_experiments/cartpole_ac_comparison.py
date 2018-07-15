"""
    This Experiment runs Sarsa-λ A2C versus Sarsa-λ Value based learning.
    All parameters on the Value Function Approximator are the same,
        but Value based learning uses epsilon-greedy and A2C uses a policy approximator NN using the A2C loss function
"""


def run_a2c_experiment(entropy_reg, run: int):
    """
    This function runs a single run of a2c on cartpole using the specified parameters
    :param entropy_reg: Entropy regularization on the policy loss function, higher means a more random policy
    :param run: Specifies the run number, this is used in the filename of the output file
    """
    import keras as ks
    import numpy as np
    from agents.actor_critic import ActorCriticAgent
    from environments.cartpole import CartPole
    from q_network_sarsa_lambda import QNetworkSL
    from p_network import PNetwork
    from experiment_util import Logger

    value_network = ks.models.Sequential()
    value_network.add(ks.layers.Dense(150, activation='relu', input_shape=(4,)))
    value_network.add(ks.layers.Dense(50, activation='relu', input_shape=(4,)))

    value_network.add(ks.layers.Dense(2, activation='linear'))

    value_network.compile(optimizer=ks.optimizers.Adam(lr=0.001),
                          loss='mse')

    policy_network = ks.models.Sequential()
    policy_network.add(ks.layers.Dense(150, activation='relu', input_shape=(4,)))
    policy_network.add(ks.layers.Dense(50, activation='relu', input_shape=(4,)))
    policy_network.add(ks.layers.Dense(2, activation='softmax'))

    l = Logger(filename="../results/AC_VS_SL_cartpole_a2c_%.5f_%d.h5"%(entropy_reg, run))
    env = CartPole(render=False)
    actions = env.valid_actions()

    dn = QNetworkSL(value_network, actions, lambda x: np.reshape(x.state, newshape=(1, 4)),
                    lambd=0.9,
                    gamma=1.0,
                    reward_factor=0.01,
                    fixed_length=100,
                    lambda_min=1e-2
                    )

    pn = PNetwork(
        policy_network,
        actions,
        lambda x: np.array(x.state),
        fixed_steps=100,
        entropy_regularization=entropy_reg,
        alpha=0.001,
        use_advantage=True
    )

    ac = ActorCriticAgent(env, dn, pn,
                          replay_memory_size=1000
                          )

    c = ac.get_configuration()
    experiment = l.start_experiment(c)
    q = ac.learn(num_episodes=250, result_handler=experiment.log)


def run_saraslambda_experiment(epsilon_start, epsilon_min, epsilon_decay, run: int):
    """
    Runs deep sarasa lambda on cartpole
    :param epsilon_start: Starting epsilon value
    :param epsilon_min: Minimum epsilon value
    :param epsilon_decay: Factor multiplied with epsilon each step
    :param run: Run identifier used in the output filename
    :return:
    """

    import keras as ks
    import numpy as np
    from agents.deep_sarsa import DeepSarsa
    from environments.cartpole import CartPole
    from q_network_sarsa_lambda import QNetworkSL
    from experiment_util import Logger

    value_network = ks.models.Sequential()
    value_network.add(ks.layers.Dense(150, activation='relu', input_shape=(4,)))
    value_network.add(ks.layers.Dense(50, activation='relu', input_shape=(4,)))

    value_network.add(ks.layers.Dense(2, activation='linear'))

    value_network.compile(optimizer=ks.optimizers.Adam(lr=0.001),
                          loss='mse')

    l = Logger(filename="../results/AC_VS_SL_cartpole_sl_%.4f_%.4f_%f_%d.h5" % (epsilon_start ,epsilon_min, epsilon_decay, run))
    env = CartPole(render=False)
    actions = env.valid_actions()

    dn = QNetworkSL(value_network, actions, lambda x: np.reshape(x.state, newshape=(1, 4)),
                    lambd=0.9,
                    gamma=1.0,
                    reward_factor=0.01,
                    fixed_length=100,
                    lambda_min=1e-2
                    )

    sarsa = DeepSarsa(env, dn, replay_memory_size=1000,
                      epsilon_min=epsilon_min, epsilon_step_factor=epsilon_decay, epsilon=epsilon_start)

    c = sarsa.get_configuration()
    experiment = l.start_experiment(c)
    q = sarsa.learn(num_episodes=250, result_handler=experiment.log)


if __name__ == "__main__":
    import multiprocessing as mp

    """
    Below is a list of functions that take a run number and run the experiment.
    Each of these experiment configurations is run many times in the loop below the list.
    More experiments can be added to try different parameters.
    """

    experiments = [
        lambda r: run_a2c_experiment(0.01, r),
        lambda r: run_saraslambda_experiment(1.0, 0.00, 0.999, r),
    ]

    for experiment in experiments:
        # Change 25 here to the desired number of runs per experiment,
        #   keep in mind that every run launches a separate process
        for run in range(25):
            mp.Process(target=experiment, args=(run,)).start()