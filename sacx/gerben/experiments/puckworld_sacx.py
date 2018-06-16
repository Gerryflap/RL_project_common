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
    from sacx.multi_task_logger import PlottingMultiTaskLogger

    width, height = size = 400, 400
    env = PuckWorld(1000, size)

    def process_state(s):
        state = s.state
        out = np.zeros((10,))
        out[0] = state["player_x"]/width
        out[1] = state["player_y"]/height
        out[2] = state["player_velocity_x"]/width
        out[3] = state["player_velocity_y"]/height
        out[4] = state["good_creep_x"]/width
        out[5] = state["good_creep_y"]/height
        out[6] = state["bad_creep_x"]/width
        out[7] = state["bad_creep_y"]/height
        out[8] = ((out[0] - out[6])**2 + (out[1] - out[7])**2)**0.5
        out[9] = ((out[0] - out[4]) ** 2 + (out[1] - out[5]) ** 2) ** 0.5
        return out



    print(isinstance(env, core.FiniteActionEnvironment))
    tasks = env.get_tasks()
    actions = env.valid_actions()
    print(actions)

    def common_net(x):
        x = ks.layers.Dense(100, activation='relu')(x)
        return x

    def task_q_net(x):
        x = ks.layers.Dense(100, activation='relu')(x)
        x = ks.layers.Dense(4, activation='linear')(x)
        return x

    def task_p_net(x):
        x = ks.layers.Dense(100, activation='relu')(x)
        x = ks.layers.Dense(4, activation='softmax')(x)
        return x

    listeners = [PlottingMultiTaskLogger(tasks, 500, ['red', 'green'])]

    q_network = QNetwork((10,), actions, tasks, common_net,
                         task_q_net, process_state, gamma=0.95, alpha=0.0001, reward_scale=10, fixed_steps=100, lambd_min=1e-2, lambd=0.5)
    p_network = PolicyNetwork((10,), actions, tasks, common_net,
                         task_p_net, process_state, entropy_regularization=0.2, alpha=0.0001, fixed_steps=100)

    agent = SACU(env, q_network, p_network, tasks, num_learn=100, scheduler_period=500, listeners=listeners)

    agent.actor()
