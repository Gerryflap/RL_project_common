import gym
import algorithms.sarsa_lambda as sl
import environments.lunar_lander_state_transform as llt

g_env = gym.make('LunarLander-v2')
env = sl.GymEnvWrapper(g_env, llt.transform_state)
agent = sl.SarsaLambdaAgent(0.2, [0,1,2,3])

episodes_per_print = 100

while True:
    score = 0
    for i in range(episodes_per_print):
        score += agent.run_episode(env)
    print("Avg_score: ", score/episodes_per_print)
    print("State space size: ", len(agent.Qsa))