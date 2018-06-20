from p_network import PNetwork

if __name__ == '__main__':
    import keras as ks
    import numpy as np
    from agents.actor_critic import ActorCriticAgent
    from environments.cartpole import CartPole
    from q_network_sarsa_lambda import QNetworkSL

    from experiment_util import Logger

    l = Logger()

    lambd = [0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
    for i in range(len(lambd)):
        value_network = ks.models.Sequential()
        value_network.add(ks.layers.Dense(100, activation='relu', input_shape=(4,)))
        value_network.add(ks.layers.Dense(100, activation='relu'))
        value_network.add(ks.layers.Dense(2, activation='linear'))

        value_network.compile(optimizer=ks.optimizers.Adam(lr=0.01),
                              loss='mse')

        policy_network = ks.models.Sequential()
        policy_network.add(ks.layers.Dense(100, activation='relu', input_shape=(4,)))
        policy_network.add(ks.layers.Dense(100, activation='relu'))
        policy_network.add(ks.layers.Dense(2, activation='softmax'))

        env = CartPole(render=True)
        actions = env.valid_actions()

        dn = QNetworkSL(value_network, actions, lambda x: np.reshape(x.state, newshape=(1, 4)),
                        lambd=lambd[i],
                        gamma=0.9,
                        reward_factor=0.01,
                        fixed_length=100
                        )

        pn = PNetwork(
            policy_network,
            actions,
            lambda x: np.array(x.state),
            fixed_steps=100,
            entropy_regularization=0.1,
            alpha=0.01
        )

        a2c = ActorCriticAgent(env, dn, pn,
                               replay_memory_size=1000
                               )

        c = a2c.get_configuration()
        print(c)
        experiment = l.start_experiment(c)
        q = a2c.learn(num_episodes=200, result_handler=experiment.log)
        experiment.save_attribute("weights", value_network.get_weights())
