import gym
import old.algorithms.deep_q_learning as dq
import tensorflow as tf
import numpy as np


def linear_combination(x):
    out = tf.keras.layers.Dense(4, activation='linear')(x)
    out = out[0]
    return out


def nn(x):
    x = tf.keras.layers.Dense(100, activation='elu')(x)
    x = tf.keras.layers.Dense(100, activation='elu')(x)
    x = tf.keras.layers.Dense(4, activation='linear')(x)
    return x


def transform_s(s):
    return np.expand_dims(np.array(s), axis=0)

g_env = gym.make('LunarLander-v2')
env = dq.GymEnvWrapper(g_env, lambda s: s)
#agent = slfa.SarsaLambdaAgent(0.2, [0,1,2,3], linear_combination, (1,8), N0=10, s_transformer=transform_s)
agent = dq.DeepQAgent([0, 1, 2, 3], nn, (8,), alpha=0.001, epsilon=0.05, gamma=0.99, epsilon_step_factor=1.0, epsilon_min=0.05, replay_mem_size=10000, fixed_steps=100, reward_scale=0.01)
episodes_per_print = 10

# Get a random trajectory to view the Q-values of during training (for evaluation)
start_state = env.reset()
while not env.terminated:
    final_state, final_reward = env.step(np.random.randint(0, 3))

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
        print("Qsa final", agent.Q(final_state, sess))
