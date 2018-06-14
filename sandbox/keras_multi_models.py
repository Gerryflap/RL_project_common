import keras as ks
import numpy as np


def common_layers(x):
    x = ks.layers.Dense(10, activation='sigmoid')(x)
    return x


def own_layers(x):
    x = ks.layers.Dense(1, activation='sigmoid')(x)
    return x

models = []

inp = ks.Input((2,))
common_out = common_layers(inp)
for i in range(2):
    models.append(ks.Model(input=inp, output=own_layers(common_out)))
    models[i].compile(ks.optimizers.Adam(0.01), ks.losses.binary_crossentropy)

xs = np.array([[0,0], [0,1], [1,0], [1,1]])
ys1 = np.array([[0], [1], [1], [0]])    #XOR
ys2 = np.array([[0], [0], [0], [1]])    #AND

for i in range(1000):
    models[0].fit(xs, ys1)
    models[1].fit(xs, ys2)

print(models[0].predict(xs))
print(models[1].predict(xs))


new_models = []
new_models.append(ks.models.model_from_json(models[0].to_json()))
new_models.append(ks.models.model_from_json(models[1].to_json()))

new_models[0].set_weights(models[0].get_weights())
print(new_models[0].get_weights())
print(new_models[1].get_weights())
print(models[1].get_weights())


