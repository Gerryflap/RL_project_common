import matplotlib.pyplot as plt


class MultiTaskLogger(object):
    def log(self, trajectory, tasks):
        raise NotImplementedError


class PlottingMultiTaskLogger(MultiTaskLogger):
    def __init__(self, tasks: list, xi, colors: list):
        self.scores = dict()
        self.xi = xi

        self.tasks = tasks
        self.colors = colors

        self.xs = []
        self.plots = dict()

        for task, color in zip(tasks, colors):
            self.scores[task] = []
            self.plots[task] = plt.plot(self.scores[task], color=color)[0]

    def log(self, trajectory, performed_tasks):

        for task, l in self.scores.items():
            if len(l) > 0:
                self.xs.append(self.xs[-1] + 1)
                l.append(l[-1])
            else:
                self.xs.append(0)
                l.append(0)

        i = 0
        while i < len(performed_tasks):
            task = performed_tasks[i]
            partial_trajectory = trajectory[i*self.xi:(i+1)*self.xi]
            summed_reward = sum([e[2][task] for e in partial_trajectory])
            step_reward = summed_reward/len(partial_trajectory)
            self.scores[task][-1] = step_reward
            i += 1

        for task, color in zip(self.tasks, self.colors):
            del self.plots[task]
            self.plots[task] = plt.plot(self.scores[task], color=color)[0]
        plt.pause(0.01)

