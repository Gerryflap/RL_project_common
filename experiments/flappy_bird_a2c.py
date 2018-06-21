from agents.actor_critic import ActorCriticAgent
from p_network import PNetwork

if __name__ == '__main__':
    import keras as ks
    import numpy as np
    from agents.deep_sarsa import DeepSarsa
    from environments.flappybird import FlappyBird
    from q_network_sarsa_lambda import QNetworkSL

    neural_network = ks.models.Sequential()
    neural_network.add(ks.layers.Dense(150, activation='relu', input_shape=(8,)))
    neural_network.add(ks.layers.Dense(50, activation='relu'))
    neural_network.add(ks.layers.Dense(2, activation='linear'))

    neural_network.compile(optimizer=ks.optimizers.Adam(lr=0.001),
                           loss='mse')

    policy_network = ks.models.Sequential()
    policy_network.add(ks.layers.Dense(150, activation='relu', input_shape=(8,)))
    policy_network.add(ks.layers.Dense(50, activation='relu'))
    policy_network.add(ks.layers.Dense(2, activation='softmax'))

    width, height = size = (288, 512)
    env = FlappyBird(size)
    actions = env.valid_actions()


    def normalize_state(s):
        o = np.zeros(shape=(1, 8))
        o[0, 0] = s.state['player_y'] / height
        o[0, 1] = s.state['player_vel']
        o[0, 2] = s.state['next_pipe_dist_to_player'] / width
        o[0, 3] = s.state['next_pipe_top_y'] / (height / 2)
        o[0, 4] = s.state['next_pipe_bottom_y'] / (height / 2)
        o[0, 5] = s.state['next_next_pipe_dist_to_player'] / width
        o[0, 6] = s.state['next_next_pipe_top_y'] / (height / 2)
        o[0, 7] = s.state['next_next_pipe_bottom_y'] / (height / 2)
        return o


    vn = QNetworkSL(neural_network, actions, normalize_state,
                    lambd=0.9,
                    gamma=0.9,
                    reward_factor=1,
                    fixed_length=100,
                    lambda_min=1e-2
                    )

    pn = PNetwork(policy_network, actions, lambda x: normalize_state(x)[0],
                  fixed_steps=100,
                  entropy_regularization=0.1,
                  alpha=0.001,
                  use_advantage=True
                  )

    dql = ActorCriticAgent(env, vn, pn, replay_memory_size=1000)

    q = dql.learn()
