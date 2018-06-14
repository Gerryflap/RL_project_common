import copy
import random
from collections import deque

from core import FiniteActionEnvironment
from policy import Policy
from q_table import QTable
from sacx.extcore import TaskEnvironment
from sacx.tasked_p_network import PolicyNetwork
from sacx.tasked_q_network import QNetwork


class SACQ:

    def __init__(self,
                 env,
                 qmodel: QNetwork,
                 amodel: PolicyNetwork,
                 gamma: float=1.0,
                 num_learn: int=100000,  # TODO -- proper default value
                 num_trajectories: int=100000,  # TODO -- proper default value
                 steps_per_episode: int=1000,  # TODO -- proper default value
                 scheduler_period: int=150,   # TODO -- proper default value
                 num_task_switches: int=10,  # TODO -- proper default value
                 num_avg_gradient: int=10,  # TODO -- proper default value
                 ):
        assert isinstance(env, TaskEnvironment) and \
               isinstance(env, FiniteActionEnvironment)

        self.env = env
        self.aux_env = copy.deepcopy(env)
        self.gamma = gamma

        self.N_learn = num_learn
        self.N_trajectories = num_trajectories
        self.num_avg_gradient = num_avg_gradient

        self.steps_per_episode = steps_per_episode
        self.scheduler_period = scheduler_period
        self.number_of_task_switches = num_task_switches

        self.qtable = QTable()
        self.replay_buffer = deque(maxlen=3000)

        self.qmodel = qmodel
        self.amodel = amodel

        self.scheduler = self.qtable.derive_policy(Policy,
                                                   lambda s: self.env.get_tasks
                                                   )

    def actor(self):

        xi = self.scheduler_period                      # Number of steps between task re-scheduling
        T = self.steps_per_episode                      # Total number of steps in an episode
        H = self.number_of_task_switches                # Number of task switches performed during an episode
        Q = self.qtable                                 # Q-table for sampling tasks from the scheduler
        M_Tau = [0] * H                                 # Number of Monte Carlo samples obtained for each task slot
        P_S = self.scheduler                            # Task scheduler
        R_M = self._return                              # Return w.r.t. the main task
        B = self.replay_buffer                          # Replay buffer to store trajectories

        pi_theta = self.amodel

        for N in range(self.N_trajectories):            # Collect new trajectory from the environment
            s = self.env.reset()                        # Obtain the initial environment state
            tau = []                                    # Store trajectory as list of (state, action)-tuples
            Tau = []                                    # Store tasks that have been scheduled
            h = 0                                       # Keep track of how many tasks have been scheduled so far
            for t in range(T):                          # Repeat for T time steps
                if t % xi == 0:                         # Check if a new task should be scheduled
                    task = P_S.sample(tuple(Tau))       # If so, sample a new task from the scheduler
                    Tau.append(task)
                    h += 1                              # Update number of tasks scheduled
                a, dist = pi_theta.\
                    sample_distribution(s, Tau[-1])     # Sample action according to latest task
                s_p, rs = self.env.step(a)              # Execute action, obtain observation and rewards
                tau.append((s, a, rs, dist))            # Add to trajectory
            # TODO -- send tau and Tau0:H to learner
            B.append((tau, Tau))                        # Add trajectory and scheduling choices to replay buffer

            for h in range(H):
                M_Tau[h] += 1
                Q[tuple(Tau[:h]), Tau[h]] += (R_M(Tau[h:]) - Q[tuple(Tau[:h]), Tau[h]]) / M_Tau[h]

    def learner(self):
        for N in range(self.N_learn):
            for k in range(1000):
                tau, Tau = self.sample_trajectory()

                # Compute gradients for policy and Q  TODO

                # Send deltas to parameter server  TODO

                # Fetch new parameters phi theta

            pass  # TODO

    def sample_trajectory(self):
        return self.replay_buffer[random.randint(0, len(self.replay_buffer) - 1)]

    def sample_task(self, tasks_scheduled):
        return self.scheduler.sample(tasks_scheduled)  # TODO -- to use or not to use

    def _return(self, tasks, task=None):
        task = task or self.env.MAIN_TASK
        r = 0
        s = self.aux_env.reset()
        for h in range(self.number_of_task_switches):
            t = h * self.steps_per_episode
            while t <= (h + 1) * self.steps_per_episode - 1:
                a = self.amodel.sample(s, tasks[h])
                s, rs = self.aux_env.step(a)
                r += self.gamma ** t * rs[task]
        return r
