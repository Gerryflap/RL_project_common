import core
from sacx.sacq import SACQ

if __name__ == '__main__':
    import keras as ks
    import numpy as np
    from sacx.puckworld import PuckWorld
    from sacx.tasked_q_network import QNetwork
    from sacx.tasked_p_network import PolicyNetwork
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

    listeners = [PlottingMultiTaskLogger(tasks, 100, ['red', 'blue','blue','blue','blue', 'green'])]

    q_network = QNetwork((10,), actions, tasks, common_net,
                         task_q_net, process_state, gamma=0.9, alpha=0.001, reward_scale=10, fixed_steps=100, lambd_min=1e-2, lambd=0.8)
    p_network = PolicyNetwork((10,), actions, tasks, common_net,
                         task_p_net, process_state, entropy_regularization=0.05, alpha=0.0001, fixed_steps=100)

    agent = SACQ(env, q_network, p_network, tasks, num_learn=10, scheduler_period=100, listeners=listeners, temperature=0.001)

    agent.learn()
