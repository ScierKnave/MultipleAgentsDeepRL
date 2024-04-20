from functools import reduce
from operator import mul
from gymnasium import spaces
import matplotlib.pyplot as plt 
import json
import os
def get_input_size(space):

    if type(space) == spaces.Box:
        return reduce(mul, space.shape)
    
    return space[0].n


class Logger:
    def __init__(self, directory):
        self.path = directory
        self.logs = {}
        self.fig = plt.figure(figsize=(10, 5))  
        plt.ion()  # Turn on interactive mode
        os.makedirs(self.path, exist_ok=True)


    def log(self, entry, value):
        if entry not in self.logs: 
            self.logs[entry] = [value]
        else: self.logs[entry].append(value)

    def update_plots(self):
        num_plots = len(self.logs)
        if len(self.fig.get_axes()) != num_plots:
            self.fig.clf()  # Clear the figure to reconfigure subplots
            self.fig.set_size_inches(10, 3 * num_plots, forward=True)  # Adjust the figure size based on number of plots
            self.ax = self.fig.subplots(nrows=num_plots, ncols=1)

        ax_array = self.ax.flatten() if num_plots > 1 else [self.ax]

        for ax, (key, data) in zip(ax_array, self.logs.items()):
            ax.clear()  
            ax.plot(data, label=f'{key}')
            ax.set_title(f"{key.capitalize()} Over Time")
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Value")
            ax.legend(loc='upper left')

        plt.tight_layout()
        plt.draw()
        plt.savefig(f"{self.path}/plots.png")




    def save_log(self):
        with open(self.path+'log.json', 'w') as file:
            json.dump(self.logs, file, indent=4)


