import h5py
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    with h5py.File("results.h5", "r") as f:
        for session_name, session in f.items():
            fig, ax = plt.subplots( nrows=1, ncols=1 )
            for result_name, result in session.items():
                print(result.attrs['configuration'])
                plt.plot(result.value)
            fig.savefig("plot1_%s.png" % session_name)
            plt.close(fig)
