import argparse
import random

import numpy as np
import torch

from env import make_env
from model import QNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/best.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--record", action="store_true",
                        help="save videos to ./videos instead of opening a window")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.record:
        from gymnasium.wrappers import RecordVideo
        env = make_env(render_mode="rgb_array")
        env = RecordVideo(env, video_folder="videos", episode_trigger=lambda e: True)
    else:
        env = make_env(render_mode="human")

    q = QNet(env.action_space.n).to(device)
    q.load_state_dict(torch.load(args.model, map_location=device))
    q.eval()

    for ep in range(args.episodes):
        state, _ = env.reset()
        done, score = False, 0.0
        while not done:
            if random.random() < 0.02:     # tiny randomness so it can't freeze forever
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.as_tensor(np.asarray(state), device=device).unsqueeze(0)
                    action = q(s).argmax(dim=1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            done = terminated or truncated
        print(f"Game {ep + 1}: score {score:.0f}")

    env.close()


if __name__ == "__main__":
    main()