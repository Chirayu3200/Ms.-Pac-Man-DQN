import csv

import matplotlib.pyplot as plt
import numpy as np

plt.ion()
fig, ax = plt.subplots(figsize=(9, 5))
WINDOW = 20

while plt.fignum_exists(fig.number):
    steps, scores = [], []
    try:
        with open("logs/scores.csv") as f:
            for row in csv.DictReader(f):
                try:
                    steps.append(int(row["step"]))
                    scores.append(float(row["score"]))
                except (TypeError, ValueError, KeyError):
                    pass   # skip a half-written last line
    except FileNotFoundError:
        pass

    ax.clear()
    if scores:
        ax.plot(steps, scores, alpha=0.3, label="each game")
        if len(scores) >= WINDOW:
            avg = np.convolve(scores, np.ones(WINDOW) / WINDOW, mode="valid")
            ax.plot(steps[WINDOW - 1:], avg, linewidth=2, label=f"{WINDOW}-game average")
        ax.legend()
    ax.set_xlabel("training steps")
    ax.set_ylabel("game score")
    ax.set_title("Live training progress")
    ax.grid(alpha=0.3)
    plt.pause(5)   # waits 5 seconds while keeping the window responsive