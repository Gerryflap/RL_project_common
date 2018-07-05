from agents.sarsalambda import SarsaLambda
from environments.simple_env import SimpleEnv

env = SimpleEnv()
procedure = SarsaLambda(env, lam=1.0, gamma=1.0, fex=lambda s: s.n, epsilon=1.0, epsilon_step_factor=0.9995, epsilon_min=0)

while True:
    q = procedure.learn(num_iter=100)
    print("Q: ", procedure.q_table.Q(0, env.action_space()[1]), " Eps: ", procedure.epsilon_v)