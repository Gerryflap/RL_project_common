
if __name__ == '__main__':
    import keras as ks
    import numpy as np
    from agents.sarsalambda import SarsaLambda
    from environments.cartpole import CartPole

    from experiment_util import Logger
    l = Logger()

    lambd = [0.9, 0.8, 0.7, 0.6, 0.5]
    for i in range(5):
        
        neural_network.compile(optimizer=ks.optimizers.Adam(lr=0.001),
                               loss='mse')
        
        env = CartPole(render=True)
        actions = env.valid_actions()
        

        
        sl = SarsaLambda(env, lam=lambd)


        c = sl.get_configuration()
        print(c)
        experiment = l.start_experiment( c )
        pi = sl.learn( num_iter=1000, result_handler=experiment.log)
        experiment.save_attribute("pi", pi)
