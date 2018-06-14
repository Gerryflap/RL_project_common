
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


    agent.actor()
