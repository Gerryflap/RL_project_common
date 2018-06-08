"""
    This is a test to see whether keras models can be used as tensorflow graphs
"""

import numpy as np
import tensorflow as tf
ks = tf.keras


model = ks.models.Sequential()
model.add(ks.layers.Dense(10, activation='relu', input_shape=(2,)))
model.add(ks.layers.Dense(5, activation='tanh'))

x = tf.placeholder(tf.float32, shape=(None, 2))

m_params = model.to_json()
print(model.get_weights())
model.set_weights(model.get_weights())
with tf.Session() as sess:
    print(tf.trainable_variables())

    with tf.variable_scope("banaan"):
        m2 = ks.models.model_from_json(m_params)
        y = m2(x)
    optimize_op = tf.train.AdamOptimizer(0.0001).minimize(y)
    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run(y, feed_dict={x:[[1,2]]}))
    for i in range(100):
        sess.run(optimize_op, feed_dict={x:[[1,2]]})
    print(sess.run(y, feed_dict={x: [[1, 2]]}))
    print(tf.trainable_variables())

