import core

if __name__ == '__main__':
    import keras as ks
    from sacx.environments.mock_env import MockEnv
    from sacx.tasked_q_network import QNetwork
    from sacx.tasked_p_network import PolicyNetwork
    from sacx.sacu import SACU

    env = MockEnv()

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


    q_network = QNetwork((5,), actions, tasks, common_net,
                         task_q_net, lambda s: s.state, gamma=0.0, alpha=0.001, reward_scale=1, fixed_steps=3)
    p_network = PolicyNetwork((5,), actions, tasks, common_net,
                         task_p_net, lambda s: s.state, entropy_regularization=0.1, alpha=0.001, fixed_steps=3)

    agent = SACU(env, q_network, p_network, tasks, num_learn=1, scheduler_period=100)

    agent.learn()
