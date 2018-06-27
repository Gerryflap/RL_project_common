from multiprocessing import Pool, TimeoutError
import numpy as np
import time
import os


def experiment(runs, episodes, sigmas, lambda_parameter):
    import numpy as np
    from agents.sarsalambda import SarsaLambda
    from environments.cartpole import NoisyCartPole


    from experiment_util import Logger

    filename = ("../results/cartpole_sarsalambda_lambda_%1.2f.h5" %lambda_parameter)
    l = Logger(filename= filename)

    def cartpole_discretization(x):
        s = x.state
        if s.shape != (4,):
            raise ValueError("Expected array of shape (4,). Instead got: %s"%(str(x.shape)))
        
        out = np.zeros((4,))

        # cart position range [-2.4 2.4]
        out[0] = np.round( 20 * (s[0]+2.4)/4.8, 0) # normalized to [0 1]
        
        # cart velocity [-inf +inf]
        out[1] = np.round( 10 * np.sqrt(np.abs(s[1])) * np.sign(s[1]), 0)
        # pole angle : [-.26rad .26rad]
        out[2] = np.round( 10 * (s[2]+.26)/.52, 0) # normalized to [0 1] then multiplied by 10
        # pole velocity at tip
        out[3] = np.round( 10 * np.sqrt(np.abs(s[3])) * np.sign(s[3]), 0)
        return tuple(out)

    def transform_state(s):
        s = s.state
        s *= np.array([1, 1, 10, 1])
        s *= 2
        s = np.round(s)
        return str(s)
    
    for run_n in range(runs):
        for sigma in sigmas:
            
            env = NoisyCartPole(std= sigma, render=False)
            actions = env.valid_actions()

            sl = SarsaLambda(env,
                              lam=lambda_parameter,
                             gamma=1.0,
                             epsilon=1.0,
                             epsilon_step_factor=0.99995,
                             epsilon_min=0.0,
                             fex=transform_state
            )

            c = sl.get_configuration()
            experiment = l.start_experiment( c )
            pi = sl.learn( num_iter=episodes, result_handler=experiment.log)
            experiment.save_attribute("pi", pi)
            print("%s finished sigma=%1.2f, run=%i" % (filename, sigma, run_n) )
    return filename

if __name__ == "__main__":
    runs = 5
    episodes = 3000
    sigmas = np.array([0, 10**-2, 10**-1, 10**-0])
    lambdas = np.array([0, 0.5, 0.75, 0.9, 1])

    with Pool(processes=12) as pool:
        for i in pool.starmap(experiment, [(runs, episodes, sigmas, l) for l in lambdas]):
            print("Finished %s" % i) 
    
    
