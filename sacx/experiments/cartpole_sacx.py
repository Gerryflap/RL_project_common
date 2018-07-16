from sacx.multi_task_logger import PlottingMultiTaskLogger

if __name__ == '__main__':
    import keras as ks
    from environments.cartpole import CartPole
    from environments.wrappers.MultiTaskWrapper import MultiTaskWrapper
    from sacx.tasked_q_network import QNetwork
    from sacx.tasked_p_network import PolicyNetwork
    from sacx.sacu import SACU
    from sacx.extcore import Task

    senv = CartPole(render=True)
    tasks = [Task("MAIN_TASK")]
    env = MultiTaskWrapper(senv, lambda s, a, r, t: {tasks[0]: r}, tasks)
    actions = env.valid_actions()

    listeners = [PlottingMultiTaskLogger(tasks, 1000, ['red'])]

    q_network = QNetwork((4,), actions, tasks, ks.layers.Dense(100, activation='relu'),
                         ks.layers.Dense(2, activation='linear'), lambda x: x.state, gamma=1.0, alpha=0.001,
                         fixed_steps=100, reward_scale=0.01)
    p_network = PolicyNetwork((4,), actions, tasks, ks.layers.Dense(100, activation='relu'),
                              ks.layers.Dense(2, activation='softmax'), lambda x: x.state, entropy_regularization=0.1,
                              alpha=0.001, fixed_steps=100)

    agent = SACU(env, q_network, p_network, tasks, num_learn=100, listeners=listeners, scheduler_period=1000)

    agent.learn()