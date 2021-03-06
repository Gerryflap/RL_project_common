"""
    Implementation of the SAC-U Agent
"""


import random
from collections import deque

import numpy as np

from core import FiniteActionEnvironment
from sacx.extcore import TaskEnvironment
from sacx.tasked_p_network import PolicyNetwork
from sacx.tasked_q_network import QNetwork


class SACU(object):
    def __init__(self,
                 env,
                 qmodel: QNetwork,
                 amodel: PolicyNetwork,
                 tasks,
                 gamma: float=1.0,
                 num_learn: int=10,  # TODO -- proper default value
                 steps_per_episode: int=1000,  # TODO -- proper default value
                 scheduler_period: int=150,   # TODO -- proper default value
                 num_avg_gradient: int=10,  # TODO -- proper default value,
                 listeners=None
                 ):

        self.listeners = listeners if listeners is not None else []
        assert isinstance(env, TaskEnvironment) and \
               isinstance(env, FiniteActionEnvironment)
        self.tasks = tasks
        self.env = env
        self.gamma = gamma

        self.N_learn = num_learn
        self.num_avg_gradient = num_avg_gradient

        self.steps_per_episode = steps_per_episode
        self.scheduler_period = scheduler_period

        self.replay_buffer = deque(maxlen=3000)

        self.qmodel = qmodel
        self.amodel = amodel

        # Set the references (this can't be done on init because both objects need each other)
        self.qmodel.p_network = amodel
        self.amodel.q_network = qmodel

    def actor(self):
        """
        Takes 1 actor step
        """

        xi = self.scheduler_period                      # Number of steps between task re-scheduling
        T = self.steps_per_episode                      # Total number of steps in an episode
        B = self.replay_buffer                          # Replay buffer to store trajectories

        pi_theta = self.amodel

        # Collect new trajectory from the environment
        s = self.env.reset()                        # Obtain the initial environment state

        for task in self.tasks:
            print(task, self.amodel.distribution(s, task), self.qmodel.Qs(s, task))

        tau = []                                    # Store trajectory as list of (state, action)-tuples
        Tau = []                                    # Store tasks that have been scheduled
        h = 0                                       # Keep track of how many tasks have been scheduled so far
        score = 0
        for t in range(T):                          # Repeat for T time steps
            if t % xi == 0:                         # Check if a new task should be scheduled
                task = self.schedule_task(Tau)      # If so, sample a new task from the scheduler
                Tau.append(task)
                print("Switching to ", task)
                h += 1                              # Update number of tasks scheduled
            a, dist = pi_theta.\
                sample_distribution(s, Tau[-1])     # Sample action according to latest task
            s_p, rs = self.env.step(a)              # Execute action, obtain observation and rewards
            tau.append((s, a, rs, dist))            # Add to trajectory
            s = s_p
            score += np.array([rs[t] for t in self.tasks])
            if s_p.is_terminal():
                break
        self._update_listeners(tau, Tau)
        print("Score: ", score)
        B.append(tau)                        # Add trajectory and scheduling choices to replay buffer

        self.train_scheduler(tau, Tau)

    def learner(self):
        """
        Takes 1 learner step
        """
        for N in range(self.N_learn):
            trajectories = self.sample_trajectories()

            # TODO: Both these methods take the full trajectories at the moment, a speedup could be achieved here
            self.qmodel.train(trajectories)
            self.amodel.train(trajectories)

    def learn(self, num_episodes=10000):
        """
        Trains the SAC-U Agent on the provided environment
        :param num_episodes: The number of episodes of training to be done
        """
        for i in range(num_episodes):
            self.actor()
            self.learner()

    def schedule_task(self, Tau):
        """
        Samples a new task from the scheduler
        :param Tau: All previous tasks
        :return: A new task
        """
        return random.choice(self.tasks)

    def sample_trajectories(self):
        """
        Samples trajectories from the replay memory
        :return: A minibatch (list) of random-length trajectories
        """
        minibatch = []
        for i in range(self.num_avg_gradient):
            trajectory = self.replay_buffer[random.randint(0, len(self.replay_buffer) - 1)]
            trajectory = trajectory[random.randint(0, len(trajectory) - 1):]
            minibatch.append(trajectory)
        return minibatch

    def _update_listeners(self, trajectory, tasks):
        """
        Updates all listeners registered to this Agent
        :param trajectory: The trajectory to be pushed to the listeners
        :param tasks: The tasks executed by the actor
        """
        for listener in self.listeners:
            listener.log(trajectory, tasks)

    def train_scheduler(self, tau, Tau):
        """
        Trains the scheduler on the passed experience
        :param tau: The trajectory
        :param Tau: The chosen actions during the generation of the trajectory
        """
        # SAC-U doesn't have a trained scheduler
        pass

