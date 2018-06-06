import gym
import algorithms.deep_sarsa_lambda_learning as dsl
import algorithms.deep_q_learning as dq
import tensorflow as tf
import numpy as np



def nn(x):
    x = tf.keras.layers.Dense(100, activation='elu')(x)
    x = tf.keras.layers.Dense(100, activation='elu')(x)
    x = tf.keras.layers.Dense(9, activation='linear')(x)
    return x


def transform_s(s):
    return np.expand_dims(np.array(s), axis=0)

g_env = gym.make('BipedalWalker-v2')
env = dq.GymEnvWrapper(g_env, lambda s: s)
action_space = [[0,0,0,0], [0,0,0,1], [0,0,0,-1],[0,0,1,0], [0,0,-1,0],[0,1,0,0],[0,-1,0,0],[1,0,0,0],[1,0,0,0]]
agent = dsl.DeepSARSALambdaAgent(1.0, action_space, nn, (24,), alpha=0.001, epsilon=0.5, gamma=1.0, epsilon_step_factor=0.999, epsilon_min=0.05, replay_mem_size=10000, fixed_steps=100, reward_scale=0.01)
episodes_per_print = 4

# Get a random trajectory to view the Q-values of during training (for evaluation)
start_state = env.reset()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    while True:
        score = 0
        for i in range(episodes_per_print):
            score += agent.run_episode(env, sess)
        agent.run_episode(env, sess, visual=True)
        print("Avg_score: ", score/episodes_per_print)
        #print("State space size: ", len(agent.Qsa))
        #print("State min: ", env.state_min)
        #print("State max: ", env.state_max)
        print("Qsa start", agent.Q(start_state, sess))
