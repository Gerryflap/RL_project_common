import os

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STARTS_WITH = "snake"

if __name__ == "__main__":
    for fname in os.listdir("../results"):
        if fname.endswith(".h5") and fname.startswith(STARTS_WITH):
            try:
                with h5py.File("../results/" + fname, "r") as f:
                    for session_name, session in f.items():
                        fig, ax = plt.subplots(nrows=1, ncols=1)
                        for result_name, result in session.items():
                            # print(result.attrs['configuration'])
                            plt.plot(result.value)
                        plt.title(fname)
                        fig.savefig(fname + "_plot.png")
                        plt.close(fig)
            except OSError:
                pass
