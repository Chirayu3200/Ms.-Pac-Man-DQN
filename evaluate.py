import argparse
import random

import numpy as np
import torch

from env import make_env
from model import QNet


def evaluate(q, device, n_games=10, seed=1000, eps=0.02):
    """Average real score over n_games, using the same seeds every time."""
    env = make_env()
    scores = []
    for i in range(n_games):
        random.seed(seed + i)
        state, _ = env.reset(seed=seed + i)
        done, score = False, 0.0
        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.as_tensor(np.asarray(state), device=device).unsqueeze(0)
                    action = q(s).argmax(dim=1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            done = terminated or truncated
        scores.append(score)
    env.close()
    return float(np.mean(scores))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/reference.pt")
    parser.add_argument("--games", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = QNet(9).to(device)
    q.load_state_dict(torch.load(args.model, map_location=device))
    q.eval()
    print(f"{args.model}: average score over {args.games} games = "
          f"{evaluate(q, device, args.games):.1f}")