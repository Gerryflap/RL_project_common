import tensorflow as tf
import numpy as np
import random
import time


class DeepQAgent(object):
    def __init__(self, action_space, deep_net, state_shape, alpha=0.001, epsilon=0.1, gamma=1.0, state_transformer=lambda s: s, epsilon_step_factor=1.0, epsilon_min=0.0, replay_mem_size=1000, fixed_steps=100, batch_size=32, reward_scale=1.0, sarsa=False):
        self.batch_in_t = tf.placeholder(shape=(None,) + state_shape, dtype=tf.float32)

        # Create the live Q-value neural network in a separate TensorFlow scope
        with tf.variable_scope("q_current"):
            self.Qsa_t = deep_net(self.batch_in_t)

        self.action_space = list(action_space)
        self.alpha_t = tf.placeholder_with_default(alpha, None)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_step_factor = epsilon_step_factor
        self.epsilon_min = epsilon_min
        self.s_transformer = state_transformer
        self.action_indices = tf.placeholder(tf.float32, shape=(None, len(action_space)))
        self.step = 0
        self.reward_scale=reward_scale

        # This deep Q-learning implementation can also be switched to SARSA if deemed necessary.
        self.sarsa = sarsa

        # Create the fixed Q-value neural network in a separate TensorFlow scope
        with tf.variable_scope("q_fixed"):
            self.Qsa_fixed_t = deep_net(self.batch_in_t)

        # The target Q-value tensor, which is used during training
        self.Q_target_t = tf.placeholder(tf.float32, shape=(None, len(action_space)))
        loss_t = (self.Qsa_t * self.action_indices - self.Q_target_t)**2

        # Apply the optimizer only on trainable variables in the live scope (not the fixed one):
        self.optimizer_t = tf.train.AdamOptimizer(self.alpha_t).minimize(loss_t, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_current'))

        self.replay_memory = []
        self.replay_mem_size = replay_mem_size
        self.fixed_steps = fixed_steps
        self.batch_size = batch_size

    def Q(self, s, sess):
        """
        Returns the Q-values of state s for every action as a numpy array.
        This is calculated using the live model parameters.
        :param s: State
        :param sess: TensorFlow session
        :return: Numpy array with a Q-value for every action in the same order as the provided action_space variable
        """
        s = self.s_transformer(s)
        return sess.run(self.Qsa_t, feed_dict= {self.batch_in_t: np.array([s])})[0]

    def Q_fixed(self, s, sess):
        """
        Returns the Q-values of state s for every action as a numpy array.
        This is calculated using the fixed model parameters.
        :param s: State
        :param sess: TensorFlow session
        :return: Numpy array with a Q-value for every action in the same order as the provided action_space variable
        """
        if type(s) == str and s == "TERMINAL":
            # Ensure that a terminal state has a fixed 0 reward, which acts like an "anchor" for the value function
            return np.zeros((len(self.action_space, )))
        s = self.s_transformer(s)
        return sess.run(self.Qsa_fixed_t, feed_dict= {self.batch_in_t: np.array([s])})[0]

    def get_action(self, s, sess):
        """
        Gets the e-greedy action
        :return: (action, action_index, q-value)
        """
        q_values = self.Q(s, sess)
        if random.random() > self.epsilon:
            # Pick greedy action
            a_i = np.argmax(q_values)
        else:
            # Pick random action
            a_i = random.randint(0, len(self.action_space)-1)
        a = self.action_space[a_i]
        q_value = q_values[a_i]
        return a, a_i, q_value

    def get_batch(self, sess):
        """
        Generates the batch by sampling randomly from the replay memory.
        :param sess: TF session
        :return: xs, q_targets, action_indices
        """
        sarsa_batch = np.array(self.replay_memory)[np.random.choice(np.arange(0, len(self.replay_memory)), self.batch_size)]

        xs = []
        q_targets = np.zeros((self.batch_size, len(self.action_space)))
        action_indices = np.zeros((self.batch_size, len(self.action_space)))
        i = 0
        for s, a_i, r, s_p, a_i_p in sarsa_batch:
            if self.sarsa:
                q_target = r + self.gamma * self.Q_fixed(s_p, sess)[a_i_p]
            else:
                q_target = r + self.gamma * np.max(self.Q_fixed(s_p, sess))
            action_indices[i, a_i] = 1
            q_targets[i, a_i] = q_target
            xs.append(self.s_transformer(s))
            i += 1
        return xs, q_targets, action_indices

    def store_experience(self, s, a_i, r, s_p, a_i_p):
        """
        Stores a SARS'A' tuple in the replay memory
        :param s: state
        :param a_i: action index (not always the action itself)
        :param r: reward
        :param s_p: state' (the next state)
        :param a_i_p: action' index (index for the next action chosen)
        :return:
        """
        self.replay_memory.append((s, a_i, r, s_p, a_i_p))
        if len(self.replay_memory) > self.replay_mem_size:
            self.replay_memory.pop(0)

    def train(self, sess):
        batch_in, q_targets, action_indices = self.get_batch(sess)
        sess.run(self.optimizer_t, feed_dict={
            self.alpha_t: self.alpha,
            self.batch_in_t: batch_in,
            self.Q_target_t: q_targets,
            self.action_indices: action_indices
        })

        self.alpha *= 1.0
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_step_factor
        elif self.epsilon_min > self.epsilon:
            self.epsilon = self.epsilon_min

        if self.step%self.fixed_steps == 0:
            self.update_fixed_q(sess)
        self.step += 1

    def run_episode(self, env, sess, visual=False):
        s = env.reset()
        a, a_i, _ = self.get_action(s, sess)
        score = 0
        env.render = visual

        while not env.terminated:
            s_p, r = env.step(a)
            a_p, a_i_p, _ = self.get_action(s_p, sess)
            if not visual:
                self.store_experience(s, a_i, r*self.reward_scale, s_p, a_i_p)
                self.train(sess)

            s, a, a_i = s_p, a_p, a_i_p
            score += r
        if not visual:
            self.store_experience(s, a_i, 0, "TERMINAL", 0)
        return score

    def update_fixed_q(self, sess):
        self._copy_from_scope("q_current", "q_fixed", sess)

    def _copy_from_scope(self, from_scope, to_scope, sess):
        scope_from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_scope)
        scope_to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=to_scope)

        from_map = dict()
        for var in scope_from_vars:
            unscoped_name = var.name[len(from_scope) + 1:]
            from_map[unscoped_name] = var

        to_map = dict()
        for var in scope_to_vars:
            unscoped_name = var.name[len(to_scope) + 1:]
            to_map[unscoped_name] = var

        assigns = []
        for varname, var in from_map.items():
            assigns.append(tf.assign(to_map[varname], var))
        sess.run(assigns)


class GymEnvWrapper(object):
    def __init__(self, gym_env, state_transformer):
        self.env = gym_env
        self.terminated = True
        self.render = False
        self.state_transformer = state_transformer

    def reset(self):
        self.terminated = False
        return self.state_transformer(self.env.reset())

    def step(self, action):
        s, r, done, _ = self.env.step(action)
        if self.render:
            self.env.render()
            time.sleep(1/60)
        self.terminated = done
        return self.state_transformer(s), r

    def set_rendering(self, value: bool):
        self.render = value


if __name__ == "__main__":
    import gym
    env = gym.make("LunarLander-v2")
    env = GymEnvWrapper(env, lambda s:s)

    def network(x):
        ks = tf.keras
        x = ks.layers.Dense(150, activation='relu')(x)
        x = ks.layers.Dense(50, activation='relu')(x)
        return ks.layers.Dense(4, activation='linear')(x)

    agent = DeepQAgent([0,1,2,3], network, alpha=0.001, state_shape=(8,), epsilon=1.0, epsilon_step_factor=0.9999, epsilon_min=0.05, gamma=1.0, fixed_steps=100, reward_scale=0.01, replay_mem_size=10000, sarsa=True)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        s = env.reset()
        while True:
            scores = []
            for i in range(4):
                score = agent.run_episode(env, sess)
                scores.append(score)
            agent.run_episode(env, sess, True)

            print("Score: ", sum(scores)/len(scores))
            print("Eps: ", agent.epsilon)
            print("Q: ", agent.Q(s, sess))
            print("Q fixed: ", agent.Q_fixed(s, sess))
