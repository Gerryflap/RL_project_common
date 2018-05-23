import gym
import algorithms.sarsa_lambda as sl
import environments.lunar_lander_state_transform as llt
import numpy as np
import threading
import multiprocessing as mup
import matplotlib.pyplot as plt


class ExperimentRunner(object):
    def __init__(self, episodes, lam=0.0, sigma=0.0):
        self.episodes = episodes
        self.lam = lam
        self.sigma = sigma
        self.thread = mup.Process(target=self.run)
        self.result = mup.Queue()

    def _run_experiment(self):
        g_env = gym.make('LunarLander-v2')
        env = sl.GymEnvWrapper(g_env, llt.get_state_transformer(self.sigma))
        agent = sl.SarsaLambdaAgent(self.lam, [0, 1, 2, 3], N0=10)
        per_episode_scores = []

        for episode in range(self.episodes):
            score = agent.run_episode(env)
            per_episode_scores.append(score)
        return np.array(per_episode_scores)

    def start(self):
        self.thread.start()

    def run(self):
        self.result.put(self._run_experiment())

    def get_result(self):
        return self.result.get()

if __name__ == "__main__":
    sigmas = np.array([0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,  0.1, 0.2, 0.5, 1.0])
    runners = []
    for sigma in sigmas:
        runner = ExperimentRunner(3000, lam=0.0, sigma=sigma)
        runners.append(runner)
        runner.start()

    results = []
    for runner in runners:
        results.append(runner.get_result())

    results = [np.mean(result[-50:]) for result in results]
    results = np.array(results)

    plt.semilogx(sigmas, results)
    plt.savefig("results.png")
