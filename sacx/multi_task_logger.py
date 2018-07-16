import matplotlib.pyplot as plt
"""
    Defines a logging framework for logging in multi task environments
"""

class MultiTaskLogger(object):
    def log(self, trajectory, tasks):
        raise NotImplementedError


class PlottingMultiTaskLogger(MultiTaskLogger):
    def __init__(self, tasks: list, xi, colors: list):
        """
        Creates a live plot for score/step for all listed tasks
        :param tasks: All tasks, assumes 0 is main
        :param xi: The scheduling interval
        :param colors: An array of plot colors for each task.
        """
        self.scores = dict()
        self.xi = xi

        self.tasks = tasks
        self.colors = colors

        self.xs = []
        self.plots = dict()

        f, (self.plt1 ,self.plt2) = plt.subplots(1, 2)

        for task, color in zip(tasks, colors):
            self.scores[task] = []
            self.plots[task] = self.plt1.plot(self.scores[task], color=color)[0]
        self.scores["main_score"] = []
        self.scores["main_score_avg"] = []
        self.main_plot = self.plt2.plot(self.scores["main_score"], color='red')[0]

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
            if len(partial_trajectory) == 0:
                continue
            step_reward = summed_reward/len(partial_trajectory)
            self.scores[task][-1] = step_reward
            i += 1

        main_score = sum([e[2][self.tasks[0]] for e in trajectory])
        self.scores["main_score"][-1] = main_score

        window = self.scores["main_score"][-10:]
        avg_score = sum(window)/len(window)
        self.scores["main_score_avg"][-1] = avg_score

        for task, color in zip(self.tasks, self.colors):
            del self.plots[task]
            self.plots[task] = self.plt1.plot(self.scores[task], color=color)[0]
        del self.main_plot
        self.main_plot = self.plt2.plot(self.scores["main_score_avg"], color='red')[0]
        plt.pause(0.01)

