"""
    This file describes a Tasked Dual Neural net.
    A tasked neural net has multiple parts:
    2 models with the same layout (hence the "dual"):
        - A common part, shared between tasks
        - A task-specific part at the end
    One of the 2 models is fixed and is synced with the live copy of the network using the sync method.
    The live copy can be fitted using the model.fit method.
"""
import keras as ks


class TaskedDualNeuralNet(object):
    def __init__(self, input_shape, shared_layers, task_specific_layers, compile_function, tasks):
        self.input_shape = input_shape
        self.tasks = tasks
        self.compile_function = compile_function
        self.task_specific_layers = task_specific_layers
        self.shared_layers = shared_layers
        self.live_models = dict()
        self.fixed_models = dict()

        self.init_models(shared_layers, task_specific_layers)

    def init_models(self, shared_layers, task_specific_layers):
        inp = ks.Input(self.input_shape)
        shared_live_out = shared_layers(inp)
        shared_fixed_out = shared_layers(inp)

        for task in self.tasks:
            self.live_models[task] = ks.Model(input=inp, output=task_specific_layers(shared_live_out))
            self.fixed_models[task] = ks.Model(input=inp, output=task_specific_layers(shared_fixed_out))
            self.compile_function(self.live_models[task])

    def predict(self, x, task, live=True):
        if live:
            return self.live_models[task].predict(x)
        else:
            return self.fixed_models[task].predict(x)

    def fit(self, x, y, task, epochs=1, verbose=False, batch_size=32):
        self.live_models[task].fit(x, y, epochs=epochs, verbose=verbose, batch_size=batch_size)

    def sync(self):
        for task in self.tasks:
            self.fixed_models[task].set_weights(self.live_models[task].get_weights())
