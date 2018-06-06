"""
    Implementation of Sarsa(λ)
"""
import random
import numpy as np
import tensorflow as tf


class SarsaLambdaAgent(object):
    def __init__(self, l, action_space, function, s_shape , gamma=1, N0 = 100, s_transformer= lambda s: s, alpha=0.01, epsilon=None):
        self.l = l
        self.input = tf.placeholder(shape=s_shape, dtype=tf.float32)
        self.Qs = function(self.input)
        self.action = tf.placeholder(tf.int32)
        self.Qsa = self.Qs[self.action]
        self.Nsa = dict()
        self.Ns = dict()
        self.action_space = action_space
        self.N0 = N0
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q_target = tf.placeholder(tf.float32)
        loss = (self.Qsa - self.Q_target)**2
        self.alpha = tf.placeholder_with_default(alpha, None)
        self.optimizer = tf.train.AdamOptimizer(self.alpha).minimize(loss)

        self.s_transformer = s_transformer

    def Q(self, s, a, sess):
        inp = self.s_transformer(s)
        return sess.run(self.Qsa, feed_dict={self.input: inp, self.action:a})

    def Qs(self, s, sess):
        inp = self.s_transformer(s)
        return sess.run(self.Q, feed_dict={self.input: inp})

    def N(self, s, a=None):
        if a is None:
            return self.Ns.get(s, 0)
        else:
            return self.Nsa.get((s,a), 0)

    def count_N(self, s, a):
        self.Nsa[(s,a)] = self.Nsa.get((s,a), 0) + 1
        self.Ns[s] = self.Nsa.get(s, 0) + 1

    def get_e_greedy(self, s, sess):
        if self.epsilon is not None:
            e = self.epsilon
        else:
            e = self.N0 / (self.N0 + self.N(s))
        if random.random() > e:
            # Greedy action:
            max_a = None
            max_Q = None
            for a, Q in zip(self.action_space, self.Qs(s, sess)):
                if max_a is None or max_Q < Q:
                    max_a = a
                    max_Q = Q
            return max_a
        else:
            return random.choice(list(self.action_space))

    def run_episode(self, env, sess):
        s = env.reset()
        a = self.get_e_greedy(s, sess)

        E = dict()
        score = 0

        while not env.terminated:
            s_prime, r = env.step(a)
            a_prime = self.get_e_greedy(s_prime, sess)
            delta = r + self.gamma * self.Q(s_prime, a_prime, sess) - self.Q(s, a, sess)
            E[(s,a)] = E.get((s,a), 0) + 12
            self.count_N(s,a)
            for s,a in E.keys():
                Q_target = self.Q(s,a, sess) + delta*E[(s,a)]
                sess.run(self.optimizer, feed_dict={self.Q_target: Q_target, self.input: self.s_transformer(s), self.action: a})
                E[(s,a)] *= self.gamma * self.l
            s, a = s_prime, a_prime
            score += r
        return score