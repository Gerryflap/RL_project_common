"""
    This Experiment runs Sarsa-λ A2C versus Sarsa-λ Value based learning.
    All parameters on the Value Function Approximator are the same,
        but Value based learning uses epsilon-greedy and A2C uses a policy approximator NN using the A2C loss function
"""


def run_a2c_experiment(entropy_reg, run):
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
    q = ac.learn(num_episodes=100, result_handler=experiment.log)


def run_saraslambda_experiment(epsilon_start, epsilon_min, epsilon_decay, run):

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
    q = sarsa.learn(num_episodes=100, result_handler=experiment.log)


if __name__ == "__main__":
    import multiprocessing as mp
    experiments = [
        lambda r: run_a2c_experiment(0.01, r),
        lambda r: run_saraslambda_experiment(1.0, 0.00, 0.999, r),
    ]

    for experiment in experiments:
        for run in range(5):
            mp.Process(target=experiment, args=(run,)).start()