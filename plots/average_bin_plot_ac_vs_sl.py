import os
from collections import defaultdict

import h5py
import json
import numpy as np
import matplotlib.pyplot as plt

STARTS_WITH = "AC_VS_SL"
BIN_SIZE = 10

if __name__ == "__main__":

    runs = defaultdict(lambda: [])

    for fname in os.listdir("../results"):
        if fname.endswith(".h5") and fname.startswith(STARTS_WITH):

            with h5py.File("../results/" + fname, "r") as f:
                for session_name, session in f.items():
                    #fig, ax = plt.subplots(nrows=1, ncols=1)
                    for result_name, result in session.items():
                        config = json.loads(result.attrs['configuration'])
                        print(config)
                        a2c = config["self"] == 'ActorCriticAgent'
                        runs[a2c].append(result.value)
                        #plt.plot(result.value)
                    #plt.title(fname)
                    #fig.savefig(fname + "_plot.png")
                    #plt.close(fig)

    a2c_vals = [False, True]
    a2c_colors = ["#0000FF", "#FF9900"]

    # fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.title("Average scores for Deep SARSA(λ) and A2C(λ)")
    for a2c, color in zip(a2c_vals, a2c_colors):
        results = runs[a2c]
        merged = np.stack(results, axis=1)
        merged = np.reshape(merged, (-1, BIN_SIZE, 5))
        merged_2 = np.reshape(merged, (-1, BIN_SIZE * 5))
        merged = np.mean(merged, axis=1)
        stds = np.std(merged, axis=1, ddof=1)/np.sqrt(5)
        # 50% confidence interval
        stds *= 0.68
        print(stds)
        means = np.mean(merged, axis=1)

        #sorted = np.sort(merged, axis=1)
        #sorted_2 = np.sort(merged_2, axis=1)
        #upper = sorted_2[:, 5]
        #lower = sorted_2[:, -6]

        upper = means + stds
        lower = means - stds

        xs = np.arange(0, 250, BIN_SIZE)

        plt.fill_between(xs, upper, lower, color=color+"55")
        plt.plot(xs, means, color=color+"FF")
    plt.legend(["SARSA(λ)", "A2C(λ)"], title="Agent")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.savefig("../plots/plot_sl_vs_ac.png", dpi=300)
    plt.close()



