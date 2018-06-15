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


    q_network = QNetwork((8,), actions, tasks, common_net,
                         task_q_net, process_state, gamma=0.9, alpha=0.01, reward_scale=1, fixed_steps=10, lambd_min=1e-2, lambd=0.9)
    p_network = PolicyNetwork((8,), actions, tasks, common_net,
                         task_p_net, process_state, entropy_regularization=0.03, alpha=0.001, fixed_steps=10)

    agent = SACU(env, q_network, p_network, tasks, num_learn=10, scheduler_period=500)

    agent.actor()
