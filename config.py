# ---------- Game ----------
ENV_ID = "ALE/MsPacman-v5"
SEED = 42

# ---------- Training length ----------
TOTAL_STEPS = 2_000_000       # agent decisions (each = 4 game frames)
WARMUP_STEPS = 20_000         # play randomly this long before learning starts
LEARN_EVERY = 4               # one learning update per 4 steps
TARGET_UPDATE_EVERY = 10_000  # refresh the frozen target network

# ---------- Exploration (epsilon-greedy) ----------
EPS_START = 1.0               # 100% random moves at the start
EPS_END = 0.05                # 5% random moves at the end
EPS_DECAY_STEPS = 1_000_000   # how long the drop takes

# ---------- Learning ----------
LEARNING_RATE = 1e-4
GAMMA = 0.99                  # how much future rewards matter
BATCH_SIZE = 32
BUFFER_SIZE = 50_000          # lower to 20_000 if you have under 8 GB RAM

# ---------- Logging ----------
LOG_EVERY = 10_000
CHECKPOINT_EVERY = 100_000
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"