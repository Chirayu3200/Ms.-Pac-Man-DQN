import csv

import matplotlib.pyplot as plt
import numpy as np

steps, scores = [], []
with open("logs/scores.csv") as f:
    for row in csv.DictReader(f):
        steps.append(int(row["step"]))
        scores.append(float(row["score"]))

scores = np.array(scores)
window = 20

plt.figure(figsize=(9, 5))
plt.plot(steps, scores, alpha=0.3, label="each game")
if len(scores) >= window:
    avg = np.convolve(scores, np.ones(window) / window, mode="valid")
    plt.plot(steps[window - 1:], avg, linewidth=2, label=f"{window}-game average")
plt.xlabel("training steps")
plt.ylabel("game score")
plt.title("Ms. Pac-Man DQN learning curve")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("logs/training_curve.png", dpi=120)
plt.show()