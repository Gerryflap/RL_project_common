import numpy as np
import time
import algorithms.deep_sarsa_lambda_learning as dsl
from environments.ron.core import EnvironmentWrapper
import tensorflow as tf
import environments.ron.pixelcopter as pxc
ks = tf.keras


def normalize_state(s):
    return np.reshape(s.observation / 256, newshape=size + (1,))

width, height = size = 256//8, 256//8
env = EnvironmentWrapper(pxc.VisualPixelCopter(size), state_transformer=normalize_state)


def network(x):
    x = ks.layers.Conv2D(filters=5, kernel_size=(5, 5), activation='relu', input_shape=size + (1,))(x)
    x = ks.layers.MaxPool2D((2,2))(x)
    x = ks.layers.Conv2D(filters=5, kernel_size=(5, 5), activation='relu')(x)
    x = ks.layers.MaxPool2D((2, 2))(x)
    x = ks.layers.Conv2D(filters=5, kernel_size=(5, 5), activation='relu')(x)
    x = ks.layers.Flatten()(x)
    x = ks.layers.Dense(units=16, activation='relu')(x)
    x = ks.layers.Dense(units=2,  activation='linear')(x)
    return x


agent = dsl.DeepSARSALambdaAgent(0.5, env.action_space, network, alpha=0.001, state_shape=size + (1,), epsilon=0.1,
                                 epsilon_step_factor=0.9999, epsilon_min=0.005, gamma=1.0, fixed_steps=100,
                                 reward_scale=0.1, replay_mem_size=10000, sarsa=True)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    s = env.reset()
    while True:
        scores = []
        for i in range(4):
            score = agent.run_episode(env, sess)
            scores.append(score)

        print("Score: ", sum(scores) / len(scores))
        print("Eps: ", agent.epsilon)
        print("Q: ", agent.Q(s, sess))
        print("Q fixed: ", agent.Q_fixed(s, sess))