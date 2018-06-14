import copy
import random
import keras as ks
from collections import deque

from core import FiniteActionEnvironment, State, Action
from sacx.extcore import TaskEnvironment, Task
from sacx.gerben.tasked_p_network import PolicyNetwork
from sacx.gerben.tasked_q_network import QNetwork


class SACU:

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
                 ):

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

        xi = self.scheduler_period                      # Number of steps between task re-scheduling
        T = self.steps_per_episode                      # Total number of steps in an episode
        B = self.replay_buffer                          # Replay buffer to store trajectories

        pi_theta = self.amodel

        while True:            # Collect new trajectory from the environment
            s = self.env.reset()                        # Obtain the initial environment state
            tau = []                                    # Store trajectory as list of (state, action)-tuples
            Tau = []                                    # Store tasks that have been scheduled
            h = 0                                       # Keep track of how many tasks have been scheduled so far
            score = 0
            for t in range(T):                          # Repeat for T time steps
                if t % xi == 0:                         # Check if a new task should be scheduled
                    task = random.choice(self.tasks)    # If so, sample a new task from the scheduler
                    Tau.append(task)
                    print("Switching to ", task)
                    h += 1                              # Update number of tasks scheduled
                a, dist = pi_theta.\
                    sample_distribution(s, Tau[-1])     # Sample action according to latest task
                s_p, rs = self.env.step(a)              # Execute action, obtain observation and rewards
                tau.append((s, a, rs, dist))            # Add to trajectory
                s = s_p
                score += rs[self.tasks[0]]
                if s_p.is_terminal():
                    break
            print("Score: ", score)
            B.append(tau)                        # Add trajectory and scheduling choices to replay buffer
            self.learner()

    def learner(self):
        for N in range(self.N_learn):
            trajectories = self.sample_trajectories()
            self.qmodel.train(trajectories)
            self.amodel.train(trajectories)

    def sample_trajectories(self):
        minibatch = []
        for i in range(self.num_avg_gradient):
            trajectory = self.replay_buffer[random.randint(0, len(self.replay_buffer) - 1)]
            trajectory = trajectory[random.randint(0, len(trajectory) - 1):]
            minibatch.append(trajectory)
        return minibatch

