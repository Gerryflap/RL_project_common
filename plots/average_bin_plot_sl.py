import os
from collections import defaultdict

import h5py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STARTS_WITH = "cartpole"
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
                        lambd, sigma = config["model"]["lambd"], config["env"]["std"]
                        runs[(lambd, sigma)].append(result.value)
                        #plt.plot(result.value)
                    #plt.title(fname)
                    #fig.savefig(fname + "_plot.png")
                    #plt.close(fig)

    sigmas = [0, 0.01, 0.1, 1]
    lambdas = [0, 0.5, 0.75, 0.9, 1]
    sigma_colors = ["#0000FF", "#FF9900", "#00AA00", "#FF0000"]

    for lambd in lambdas:
        # fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.title("Average scores for Deep SARSA(λ) with λ=%.2f"%lambd)
        for sigma, color in zip(sigmas, sigma_colors):
            results = runs[(lambd, sigma)]
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
        plt.legend(sigmas, title="Noise σ before scaling")
        plt.xlabel("Epsiode")
        plt.ylabel("Score")
        plt.savefig("../plots/plot_lambda_%.2f.png"%lambd, dpi=300)
        plt.close()



