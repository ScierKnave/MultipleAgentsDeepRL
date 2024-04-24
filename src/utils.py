from functools import reduce
from operator import mul
from gymnasium import spaces
import matplotlib.pyplot as plt 
import matplotlib.style as style
import json
import os
from collections import defaultdict



def get_input_size(space):
    if isinstance(space, spaces.Box):
        return reduce(mul, space.shape)
    elif isinstance(space, spaces.Discrete):
        return space.n
    elif isinstance(space, spaces.Tuple):
        print(space)
        return get_input_size(space[0])
    else:
        raise ValueError("Invalid space type")


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
        grouped_logs = defaultdict(dict)
        for key in self.logs:
            base_key = key.rsplit('_', 1)[0]  # Remove suffix after last underscore
            suffix = key.rsplit('_', 1)[-1] if '_' in key else ''
            grouped_logs[base_key][suffix] = self.logs[key]

        num_plots = len(grouped_logs)
        if len(self.fig.get_axes()) != num_plots:
            self.fig.clf()  # Clear the figure to reconfigure subplots
            self.fig.set_size_inches(10, 3 * num_plots, forward=True)  # Adjust the figure size based on number of plots
            self.ax = self.fig.subplots(nrows=num_plots, ncols=1)

        ax_array = self.ax if num_plots > 1 else [self.ax]


        for ax, (base_key, logs) in zip(ax_array, grouped_logs.items()):
            ax.clear()
            for suffix, data in logs.items():
                label = f'Mean {base_key} {suffix}' if suffix else base_key
                ax.plot(data, label=label)
            ax.set_title(f"{base_key.capitalize()} Over Time", fontsize=14, fontweight='bold')
            ax.set_xlabel("Iterations", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.legend(loc='upper left', fontsize=10)

        plt.tight_layout()
        plt.draw()
        plt.savefig(f"{self.path}/plots.png")




    def save_log(self):
        with open(f"{self.path}/log.json", 'w') as file:
            json.dump(self.logs, file, indent=4)


