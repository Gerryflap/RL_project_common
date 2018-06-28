
if __name__ == '__main__':
    import keras as ks
    import numpy as np
    from agents.deep_sarsa import DeepSarsa
    from environments.cartpole import CartPole
    from q_network_sarsa_lambda import QNetworkSL

    from experiment_util import Logger
    l = Logger()

    lambd = [1.0]
    for i in range(len(lambd)):
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
                         lambda_min=1e-3,
                         gamma=1.0,
                         reward_factor=0.01,
                         fixed_length=100
        )
        
        dql = DeepSarsa(env, dqn,
                        epsilon=1.0,
                        epsilon_step_factor=0.9995,
                        epsilon_min=0.0,
                        replay_memory_size=1000
        )

        c = dql.get_configuration()
        print(c)
        experiment = l.start_experiment( c )
        try:
            q = dql.learn( num_episodes=250, result_handler=experiment.log)
        except KeyboardInterrupt:
            pass
        dqn.live_model.save_weights("weights")
