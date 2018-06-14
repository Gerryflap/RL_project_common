import environments.cartpole
import environments.wrappers.standardized_env_wrapper as wrapper
import agents.deep_sarsa_lambda as dsl
import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
ks = tf.keras


neural_network = ks.models.Sequential()
neural_network.add(ks.layers.Conv2D(32, 8, strides=6, activation='relu', input_shape=(85,80,6)))
neural_network.add(ks.layers.Conv2D(64, 6, strides=4, activation='relu'))
neural_network.add(ks.layers.Flatten())
neural_network.add(ks.layers.Dense(200, activation='relu'))
neural_network.add(ks.layers.Dense(4, activation='linear'))

print(neural_network.summary())

def transform_state(x, x_old):
    return np.concatenate((x[40::2, ::2], x_old[40::2, ::2]), axis=2)


genv = gym.make('Breakout-v0')
env = dsl.GymEnvWrapper(genv, transform_state, keep_previous_frame=True)
agent = dsl.DeepSARSALambdaAgent(0.5, [0,1,2,3], neural_network, None,
                                 epsilon=0.9, epsilon_min=0.05, epsilon_step_factor=0.99999,
                                 batch_size=10,
                                 alpha=0.001, reward_scale=0.1, gamma=0.9, replay_mem_size=1000,
                                 lambda_lower_bound=0.05
                                 )

# genv.reset()
# for i in range(10):
#     genv.step(genv.action_space.sample())
# s = genv.step(genv.action_space.sample())[0]
# plt.imshow(transform_state(s))
# plt.show()

while True:
    scores = []
    for i in range(3):
        score = agent.run_episode(env, True)
        scores.append(score)
    print("Average score: ", np.mean(scores), agent.epsilon)
    #agent.run_episode(env, True)
