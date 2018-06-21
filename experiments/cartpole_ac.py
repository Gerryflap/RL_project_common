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
        value_network.add(ks.layers.Dense(150, activation='relu', input_shape=(4,)))
        value_network.add(ks.layers.Dense(50, activation='relu', input_shape=(4,)))

        value_network.add(ks.layers.Dense(2, activation='linear'))

        value_network.compile(optimizer=ks.optimizers.Adam(lr=0.001),
                              loss='mse')

        policy_network = ks.models.Sequential()
        policy_network.add(ks.layers.Dense(150, activation='relu', input_shape=(4,)))
        policy_network.add(ks.layers.Dense(50, activation='relu', input_shape=(4,)))
        policy_network.add(ks.layers.Dense(2, activation='softmax'))

        env = CartPole(render=False)
        actions = env.valid_actions()

        dn = QNetworkSL(value_network, actions, lambda x: np.reshape(x.state, newshape=(1, 4)),
                        lambd=lambd[i],
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
            entropy_regularization=0.1,
            alpha=0.001
        )

        ac = ActorCriticAgent(env, dn, pn,
                               replay_memory_size=1000
                               )

        c = ac.get_configuration()
        print(c)
        experiment = l.start_experiment(c)
        q = ac.learn(num_episodes=1000, result_handler=experiment.log)
        experiment.save_attribute("weights", value_network.get_weights())
