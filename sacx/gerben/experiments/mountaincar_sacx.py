import core

if __name__ == '__main__':
    import keras as ks
    import numpy as np
    from sacx.gerben.environments.mountaincar import MountainCar
    from environments.wrappers.MultiTaskWrapper import MultiTaskWrapper
    from sacx.gerben.tasked_q_network import QNetwork
    from sacx.gerben.tasked_p_network import PolicyNetwork
    from sacx.gerben.sacu import SACU
    from sacx.gerben.extcore import Task

    env = MountainCar()

    print(isinstance(env, core.FiniteActionEnvironment))
    tasks = env.get_tasks()
    actions = env.valid_actions()
    print(actions)

    def common_net(x):
        x = ks.layers.Dense(100, activation='relu')(x)
        return x

    def task_q_net(x):
        x = ks.layers.Dense(100, activation='relu')(x)
        x = ks.layers.Dense(3, activation='linear')(x)
        return x

    def task_p_net(x):
        x = ks.layers.Dense(100, activation='relu')(x)
        x = ks.layers.Dense(3, activation='softmax')(x)
        return x


    q_network = QNetwork((2,), actions, tasks, common_net,
                         task_q_net, lambda s: s.state,
                         gamma=0.9, alpha=0.01, reward_scale=1.0, fixed_steps=100, lambd=0.9, lambd_min=1e-4
                         )
    p_network = PolicyNetwork((2,), actions, tasks, common_net,
                         task_p_net, lambda s: s.state, entropy_regularization=0.05, alpha=0.001, fixed_steps=100)

    agent = SACU(env, q_network, p_network, tasks, num_learn=10, scheduler_period=200)

    agent.actor()
