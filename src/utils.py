from functools import reduce
from operator import mul
from gymnasium import spaces
import matplotlib.pyplot as plt 
import matplotlib.style as style
import json
import os
from collections import defaultdict
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import matplotlib.animation as animation

plt.style.use('ggplot')

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
    def __init__(self, directory, k=100):
        self.path = directory
        self.logs = {}
        self.fig = plt.figure(figsize=(10, 5))  
        os.makedirs(self.path, exist_ok=True)
        self.anim = None
        self.count = 0  # Add counter
        self.k = k  # Number of updates between each video rendering
        self.count = 0
        self.states = []
        
    def log(self, entry, value):
        if entry not in self.logs: 
            self.logs[entry] = [value]
        else: self.logs[entry].append(value)

        self.count += 1

        if self.count % self.k == 0:  # save the state every k steps
            self.states.append({key: values[:self.count//self.k+1] for key, values in self.logs.items()})
            self.render_video()
            
    def animate(self, i):
        self.update_plots(self.states[i])

    def update_plots(self, logs=None):
            if logs is None: logs = self.logs
            grouped_logs = defaultdict(dict)
            for key in logs:
                base_key = key.rsplit('_', 1)[0]
                suffix = key.rsplit('_', 1)[-1] if '_' in key else ''
                grouped_logs[base_key][suffix] = logs[key]

            num_plots = len(grouped_logs)
            if len(self.fig.get_axes()) != num_plots:
                self.fig.clf()
                self.fig.set_size_inches(10, 3 * num_plots, forward=True)
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




    def save_log(self):
        with open(f"{self.path}/log.json", 'w') as file:
            json.dump(self.logs, file, indent=4)

    def render_video(self):
        anim = animation.FuncAnimation(self.fig, 
                                       self.animate, 
                                       frames=range(len(self.states)),
                                       interval=10
                                       )
        anim.save(f"{self.path}/plots_{self.count//self.k}.mp4", writer='ffmpeg')