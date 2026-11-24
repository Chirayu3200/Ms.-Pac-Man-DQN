import argparse
import csv
import os
import random
import time
from collections import deque

import numpy as np
import torch

import config
from agent import DQNAgent
from env import make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=config.TOTAL_STEPS,
                        help="total training steps (use 100000 for a quick test)")
    args = parser.parse_args()

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    env = make_env()
    agent = DQNAgent(env.action_space.n, env.observation_space.shape, device)

    # Scale exploration/warmup down automatically for short test runs
    eps_steps = min(config.EPS_DECAY_STEPS, args.steps // 2)
    warmup = min(config.WARMUP_STEPS, args.steps // 10)

    log_file = open(os.path.join(config.LOG_DIR, "scores.csv"), "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["episode", "step", "score"])

    state, _ = env.reset(seed=config.SEED)
    score, episode = 0.0, 0
    recent = deque(maxlen=20)
    best_avg = -float("inf")
    start = time.time()

    for step in range(1, args.steps + 1):
        eps = agent.epsilon(step, eps_steps)
        action = agent.act(state, eps)
        next_state, reward, terminated, truncated, _ = env.step(action)
        score += reward                               # real game score, for reporting

        # Store a CLIPPED reward (+1, 0, or -1) so big point values don't destabilize learning.
        # 'terminated' = real game over (no future). 'truncated' = time limit (future still exists).
        agent.buffer.add(state, action, np.sign(reward), next_state, terminated)
        state = next_state

        if terminated or truncated:
            episode += 1
            recent.append(score)
            log.writerow([episode, step, score])
            log_file.flush()
            if len(recent) == recent.maxlen and np.mean(recent) > best_avg:
                best_avg = float(np.mean(recent))
                agent.save(os.path.join(config.CHECKPOINT_DIR, "best.pt"))
            score = 0.0
            state, _ = env.reset()

        if step > warmup and step % config.LEARN_EVERY == 0:
            agent.learn()

        if step % config.TARGET_UPDATE_EVERY == 0:
            agent.update_target()

        if step % config.LOG_EVERY == 0:
            avg = np.mean(recent) if recent else 0.0
            mins = (time.time() - start) / 60
            print(f"step {step:>9,} | games {episode:>5} | avg score (last 20) {avg:7.1f} "
                  f"| eps {eps:.2f} | {mins:.1f} min")

        if step % config.CHECKPOINT_EVERY == 0:
            agent.save(os.path.join(config.CHECKPOINT_DIR, "latest.pt"))

    agent.save(os.path.join(config.CHECKPOINT_DIR, "final.pt"))
    log_file.close()
    print("Done. Models saved in", config.CHECKPOINT_DIR)


if __name__ == "__main__":
    main()