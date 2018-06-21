from collections import defaultdict

from policy import GreedyPolicy
from q_network import QNetwork
from q_table import QTable
from sacx.gerben.sacu import SACU
from sacx.gerben.tasked_p_network import PolicyNetwork


class SACQ(SACU):
    def __init__(self, env, qmodel: QNetwork, amodel: PolicyNetwork, tasks, gamma: float = 1.0, num_learn: int = 10,
                 steps_per_episode: int = 1000, scheduler_period: int = 150, num_avg_gradient: int = 10,
                 listeners=None):
        super().__init__(env, qmodel, amodel, tasks, gamma, num_learn, steps_per_episode, scheduler_period,
                         num_avg_gradient, listeners)
        self.Q = QTable()
        self.M = defaultdict(lambda: 0)
        self.scheduler = self.Q.derive_policy(GreedyPolicy, lambda x: self.tasks)


    def train_scheduler(self, tau, Tau):
        main_task = self.tasks[0]
        xi = self.scheduler_period
        main_rewards = [r[main_task] for _, _, r, _ in tau]
        for h in range(len(Tau)):
            R = sum(main_rewards[h*xi:])
            self.M[Tau[h]] += 1
            self.Q[tuple(Tau[:h]), Tau[h]] += (R - self.Q[tuple(Tau[:h]), Tau[h]])/self.M[Tau[h]]

    def schedule_task(self, Tau):
        return self.scheduler.sample(tuple(Tau))
