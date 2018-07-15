# Methods for Deep Reinforcement Learning

This repository contains code made for a Capita Selecta on Reinforcement Learning.

The report (and code) cover the following subjects:
- Traditional RL (Monte Carlo, SARSA-λ)
- Linear value function approximation (Linear function approx SARSA-λ)
- (Deep) Neural Network based function approximation inspired by DQN (Deep Q learning, Deep SARSA-λ)
- An Actor Critic algorithm (Using A2C with SARSA-λ as value function approximator)
- SAC-Q (https://deepmind.com/blog/learning-playing/)

These algorithms were evaluated in multiple experiments. 
The experiments used in the paper are all in the folder ```cluster_experiments```, 
apart from the SAC-Q experiment which resides in ```sacx/experiments/mountaincar.py```.

## Explanation of the folders:

### Agents
The agents folder contains all agents
