import os
from collections import defaultdict

import h5py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STARTS_WITH = "snake"
BIN_SIZE = 40
experiments = [{'legend': 'Feature Extracted', 'file': 'deep', 'color': '#0000FF'},
               {'legend': 'Raw Pixels', 'file': 'conv', 'color': '#FF0000'}]

if __name__ == "__main__":

    runs = defaultdict(lambda: [])

    for fname in os.listdir("../results"):
        if fname.endswith(".h5") and fname.startswith(STARTS_WITH):
            try:
                with h5py.File("../results/" + fname, "r") as f:
                    session_name, session = [(a, b) for a, b in f.items()][-1] # Hacky way of getting the last session
                    for result_name, result in session.items():
                        config = json.loads(result.attrs['configuration'])
                        for experiment in experiments:
                            if experiment['file'] in fname:
                                runs[experiment['file']].append(result.value)
            except OSError:
                pass

    plt.title('Average scores for Feature Extracted and Pixel Snake')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    for experiment in experiments:
        results = runs[experiment['file']]
        # make everything a short as the shortest experiment
        min_len = min([len(res) for res in results])
        min_len = min_len - min_len % BIN_SIZE
        results = list(map(lambda res: res[0:min_len], results))
        n = np.shape(results)[0]  # Extract the number of samples
        print(n)

        merged = np.stack(results, axis=1)
        merged = np.reshape(merged, (-1, BIN_SIZE, n))
        # merged_2 = np.reshape(merged, (-1, BIN_SIZE * n))
        merged = np.mean(merged, axis=1)
        stds = np.std(merged, axis=1, ddof=1) / np.sqrt(n)
        # 50% confidence interval
        stds *= 0.68
        means = np.mean(merged, axis=1)

        upper = means + stds
        lower = means - stds

        xs = np.arange(0, min_len, BIN_SIZE)

        plt.fill_between(xs, upper, lower, color=experiment['color'] + "55")
        plt.plot(xs, means, color=experiment['color'] + "FF")
    plt.legend([x['legend'] for x in experiments], title='Input space type')
    plt.savefig("../plots/snake.png", dpi=500)
    plt.close()
