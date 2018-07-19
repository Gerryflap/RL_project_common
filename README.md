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

If you wish to read and gain a more in-depth understanding of these algorithyms, we invite you to read the report or peruse the assosiated slides:
- [Report](../master/methods-deep-reinforcement.pdf)
- [Slides](../master/MethodsForDeepRL-Slides.pdf)

## Some shiny gifs
Cart Pole:

![alt-text](../master/gifs/cart_pole.gif "Cart Pole")

Snake:

![alt text](../master/gifs/snake.gif "CNN snake")

## Dependencies
In order to run the experiments a number of different Python packages are required, including Tensorflow, Keras, Numpy, Matplotlib, H5py and Pandas. Alternatively an environment that contains all dependencies can be installed using Conda:

```
conda create --name rl --file requirements.txt
```

The environment containing the dependencies can then be activated using:
```
source activate rl
```
## Environments
You may find that you need to install the environments seperately depending on which experiments you plan on running. The environments used within the experiments here are either [gym](http://gym.openai.com/) environments, [PLE](http://pygame-learning-environment.readthedocs.io/en/latest/), or the snake environment.

The gym and PLE environments can be installed with your favoriate package manager and the snake environement can be found [here](https://github.com/av80r/Gym-Snake).

 If you wish to run the agents against a different environment, this is a relatively trivial task. You need to write a wrapper like those that can be found in the `/environments` folder and supply the state and action spaces.


## Running experiments
Experiments can be run as follows:

```
python -m cluster_experiments.cartpole_sarsa_lambda
```

Results and configuration are logged to `results/<filename>.h5`, where the filename depends on the experiment. Depending on how logging is used in the experiment, the log file contains results for multiple runs and parameters of the experiment. As an example for how the log files can be read, see the scripts in `/plots`  



