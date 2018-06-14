import core

if __name__ == '__main__':
    import keras as ks
    import numpy as np
    from sacx.puckworld import PuckWorld
    from environments.wrappers.MultiTaskWrapper import MultiTaskWrapper
    from sacx.gerben.tasked_q_network import QNetwork
    from sacx.gerben.tasked_p_network import PolicyNetwork
    from sacx.gerben.sacu import SACU
    from sacx.gerben.extcore import Task

    width, height = size = 400, 400
    env = PuckWorld(1000, size)

    def process_state(s):
        state = s.state
        out = np.zeros((8,))
        out[0] = state["player_x"]/width
        out[1] = state["player_y"]/height
        out[2] = state["player_velocity_x"]/width
        out[3] = state["player_velocity_y"]/height
        out[4] = state["good_creep_x"]/width
        out[5] = state["good_creep_y"]/height
        out[6] = state["bad_creep_x"]/width
        out[7] = state["bad_creep_y"]/height
        return out



    print(isinstance(env, core.FiniteActionEnvironment))
    tasks = env.get_tasks()
    actions = env.valid_actions()
    print(actions)

    q_network = QNetwork((8,), actions, tasks, ks.layers.Dense(100, activation='relu'),
                         ks.layers.Dense(4, activation='linear'), process_state, gamma=0.99, alpha=0.001, reward_scale=0.0001)
    p_network = PolicyNetwork((8,), actions, tasks, ks.layers.Dense(100, activation='relu'),
                         ks.layers.Dense(4, activation='softmax'), process_state, entropy_regularization=3.0, alpha=0.0000001)

    agent = SACU(env, q_network, p_network, tasks, num_learn=1)

    # dqn = QNetworkSL(neural_network, actions, lambda x: np.reshape(x.state, newshape=(1, 4)),
    #                  lambd=0.9,
    #                  gamma=0.9,
    #                  reward_factor=0.01,
    #                  fixed_length=100
    #                  )
    #
    # dql = DeepSarsa(env, dqn,
    #                 epsilon=0.3,
    #                 epsilon_step_factor=0.99995,
    #                 epsilon_min=0.05,
    #                 replay_memory_size=1000
    #                 )

    agent.actor()
