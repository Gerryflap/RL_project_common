import old.algorithms.sarsa_lambda as sl
from old.environments.ron.pixelcopter import PixelCopter
import old.environments.ron.core as core
import numpy as np

width, height = 256, 256


def signed_sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def discretize_state(s):
    r = lambda x: np.round(x, 0)
    y = s.observation['player_y'] / height
    c_dist = s.observation['player_dist_to_ceil'] / (height / 2)
    f_dist = s.observation['player_dist_to_floor'] / (height / 2)
    player_vel = s.observation['player_vel']
    ng_dist = s.observation['next_gate_dist_to_player'] / width
    ng_bt = s.observation['next_gate_block_top'] / height
    ng_bb = s.observation['next_gate_block_bottom'] / height

    ng_bt_dist = r(10 * signed_sqrt(ng_bt - y))
    ng_bb_dist = r(10 * signed_sqrt(ng_bb - y))

    c_dist = r(10 * signed_sqrt(c_dist))
    f_dist = r(10 * signed_sqrt(f_dist))
    ng_dist = r(5 * signed_sqrt(ng_dist))

    player_vel = r(1 * signed_sqrt(player_vel))
    #print(ng_bb_dist, ng_bt_dist, player_vel, c_dist, f_dist, ng_dist)

    return ng_bb_dist, ng_bt_dist, player_vel, c_dist, f_dist, ng_dist


env = core.EnvironmentWrapper(PixelCopter((width, height)), state_transformer=discretize_state)
agent = sl.SarsaLambdaAgent(0.9, env.action_space, epsilon_constant=0.005)

while True:
    scores = []
    for i in range(100):
        score = agent.run_episode(env)
        scores.append(score)
    agent.run_episode(env, slow_fps=True)
    print("Average score: ", np.average(scores))
    print("Qsa size: ", len(agent.Qsa))
