import gymnasium as gym
import ale_py
import pygame

gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array",
               frameskip=1, repeat_action_probability=0.0)
env.reset()
frame = env.render()
h, w = frame.shape[:2]
ZOOM = 3

pygame.init()
screen = pygame.display.set_mode((w * ZOOM, h * ZOOM))
pygame.display.set_caption("Ms. Pac-Man (arrow keys or WASD, Esc to quit)")
clock = pygame.time.Clock()

score = 0.0
running = True
print("Window opened. Click it, then use the arrow keys or WASD.")

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    k = pygame.key.get_pressed()
    up    = k[pygame.K_UP]    or k[pygame.K_w]
    down  = k[pygame.K_DOWN]  or k[pygame.K_s]
    left  = k[pygame.K_LEFT]  or k[pygame.K_a]
    right = k[pygame.K_RIGHT] or k[pygame.K_d]

    # 0 NOOP, 1 UP, 2 RIGHT, 3 LEFT, 4 DOWN, 5 UPRIGHT, 6 UPLEFT, 7 DOWNRIGHT, 8 DOWNLEFT
    if up and right:     action = 5
    elif up and left:    action = 6
    elif down and right: action = 7
    elif down and left:  action = 8
    elif up:             action = 1
    elif right:          action = 2
    elif left:           action = 3
    elif down:           action = 4
    else:                action = 0

    _, reward, terminated, truncated, _ = env.step(action)
    score += reward

    frame = env.render()
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    surface = pygame.transform.scale(surface, (w * ZOOM, h * ZOOM))
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    if terminated or truncated:
        print(f"Game over. Your score: {score:.0f}")
        score = 0.0
        env.reset()

    clock.tick(60)   # run at the game's normal speed

env.close()
pygame.quit()