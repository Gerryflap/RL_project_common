
if __name__ == '__main__':
    import keras as ks
    import numpy as np
    from environments.cartpole import CartPole
    from environments.wrappers.MultiTaskWrapper import MultiTaskWrapper
    from sacx.gerben.tasked_q_network import QNetwork
    from sacx.gerben.tasked_p_network import PolicyNetwork
    from sacx.gerben.sacu import SACU
    from sacx.gerben.extcore import Task



    senv = CartPole(render=True)
    tasks = [Task("MAIN_TASK")]
    env = MultiTaskWrapper(senv, lambda s, a, r, t: {tasks[0]: r}, tasks)
    actions = env.valid_actions()

    q_network = QNetwork((4,), actions, tasks, ks.layers.Dense(100, activation='relu'),
                         ks.layers.Dense(2, activation='linear'), lambda x: x.state, gamma=0.9, alpha=0.001)
    p_network = PolicyNetwork((4,), actions, tasks, ks.layers.Dense(100, activation='relu'),
                         ks.layers.Dense(2, activation='softmax'), lambda x: x.state, entropy_regularization=0.3, alpha=0.0001)

    agent = SACU(env, q_network, p_network, tasks, num_learn=100)

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
