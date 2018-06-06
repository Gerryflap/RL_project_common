import gym
import algorithms.sarsa_lambda as sl
import environments.lunar_lander_state_transform as llt
import numpy as np
import multiprocessing as mup
import matplotlib
matplotlib.use("agg")
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
    sigmas = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05,  0.1, 0.5, 1.0])
    lambdas = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99, 1.0])
    runners = []
    xs = np.array([[l, s] for l in lambdas for s in sigmas])
    for batch_index in range(10):
        batch = []
        for l, s in xs:
            runner = ExperimentRunner(3000, lam=l, sigma=s)
            batch.append(runner)
            runner.start()
        runners.append(batch)

    results = []
    for batch in runners:
        batch_results = []
        for runner in batch:
            batch_results.append(runner.get_result())
        results.append(batch_results)

    np.save("results_full", np.array(results))
    results = [[np.mean(result[-100:]) for result in batch] for batch in results]
    results = np.array(results)

    results = np.mean(results, axis=0)

    np.save("xs", xs)
    np.save("results", results)

    #plt.plot(xs, results)
    #plt.savefig("results.png")