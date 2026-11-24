import gymnasium
import ale_py
import torch

from env import make_env

print("gymnasium:", gymnasium.__version__)
print("ale-py:", ale_py.__version__)
print("torch:", torch.__version__, "| GPU available:", torch.cuda.is_available())

env = make_env()
state, _ = env.reset(seed=0)
print("observation shape:", state.shape, "| dtype:", state.dtype)
print("number of actions:", env.action_space.n)

score, steps, done = 0.0, 0, False
while not done:
    state, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    score += reward
    steps += 1
    done = terminated or truncated
print(f"random player: {steps} steps, score {score:.0f}")
print("Setup OK")