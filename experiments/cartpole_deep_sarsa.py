import environments.cartpole
import environments.wrappers.standardized_env_wrapper as wrapper
import agents.deep_sarsa_lambda as dsl
import tensorflow as tf
import numpy as np


def neural_network(x):
    ks = tf.keras
    x = ks.layers.Dense(150, activation='relu')(x)
    x = ks.layers.Dense(50, activation='relu')(x)
    return ks.layers.Dense(2, activation='linear')(x)


senv = environments.cartpole.CartPole()
env = wrapper.StandardizedEnvWrapper(senv, lambda s: s.observation)
agent = dsl.DeepSARSALambdaAgent(1, env.action_space, neural_network, (4,),
                                 epsilon=0.9, epsilon_min=0.05, epsilon_step_factor=0.999,
                                 alpha=0.0001, reward_scale=0.01, gamma=0.9)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    while True:
        scores = []
        for i in range(5):
            score = agent.run_episode(env, sess)
            scores.append(score)
        print("Average score: ", np.mean(scores))
