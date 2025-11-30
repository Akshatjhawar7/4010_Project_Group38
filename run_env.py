import gym
import pygame
import numpy as np
import time

# Importing env so it gets registered
from TrafficRLEnv import TrafficIntersectionEnv

# Constants for visuals
WIDTH, HEIGHT = 500, 500
BG_COLOR = (30, 30, 30)
CAR_COLOR = (0, 0, 255)
PED_COLOR = (255, 255, 0)
FONT_COLOR = (255, 255, 255)

def draw_env(screen, env):
    screen.fill((30, 30, 30))  # Background

    font = pygame.font.SysFont(None, 24)
    phase_str = ['GREEN_NS', 'YELLOW_NS', 'GREEN_EW', 'YELLOW_EW'][env.phase]
    text = font.render(f"Phase: {phase_str}, Timer: {env.phase_timer}", True, (255, 255, 255))
    screen.blit(text, (10, 10))

    center = WIDTH // 2
    road_w = 60
    stripe_w = 6
    stripe_gap = 6
    stripe_len = 20
    crosswalk_len = 40

    # Drawing roads
    pygame.draw.rect(screen, (100, 100, 100), (center - road_w//2, 0, road_w, HEIGHT))  # vertical
    pygame.draw.rect(screen, (100, 100, 100), (0, center - road_w//2, WIDTH, road_w))  # horizontal

    # --- Signal Lights ---
    signal_radius = 8
    signal_offset = 30

    signal_positions = [
        (center - signal_offset, center - road_w // 2 - 30),  # North signal
        (center + road_w // 2 + 30, center - signal_offset),  # East signal
        (center + signal_offset, center + road_w // 2 + 30),  # South signal
        (center - road_w // 2 - 30, center + signal_offset)   # West signal
    ]

    # Defining which directions have green in each phase
    phase_green = {
        0: [0, 2],  # GREEN_NS
        1: [],      # YELLOW_NS (transitional)
        2: [1, 3],  # GREEN_EW
        3: []       # YELLOW_EW (transitional)
    }

    # Drawing signals
    for i, (x, y) in enumerate(signal_positions):
        if i in phase_green.get(env.phase, []):
            # Drawing arrow for green direction
            if i == 0:  # North
                arrow = [(x, y - 10), (x - 5, y), (x + 5, y)]
            elif i == 1:  # East
                arrow = [(x + 10, y), (x, y - 5), (x, y + 5)]
            elif i == 2:  # South
                arrow = [(x, y + 10), (x - 5, y), (x + 5, y)]
            elif i == 3:  # West
                arrow = [(x - 10, y), (x, y - 5), (x, y + 5)]
            pygame.draw.polygon(screen, (0, 255, 0), arrow)
        else:
            pygame.draw.circle(screen, (200, 0, 0), (x, y), signal_radius)

    # Crosswalks - horizontal
    # Drawing crosswalk stripes
    for i in range(-2, 3):
        offset = i * (stripe_w + stripe_gap)
    
        # Vertical crosswalks (North and South)
        pygame.draw.rect(screen, (220, 220, 220), (center - stripe_w//2 + offset, center - road_w//2 - crosswalk_len + 20, stripe_w, stripe_len))  # North
        pygame.draw.rect(screen, (220, 220, 220), (center - stripe_w//2 + offset, center + road_w//2, stripe_w, stripe_len))  # South

        # Horizontal crosswalks (West and East)
        pygame.draw.rect(screen, (220, 220, 220), (center - road_w//2 - crosswalk_len + 20, center - stripe_w//2 + offset, stripe_len, stripe_w))  # West
        pygame.draw.rect(screen, (220, 220, 220), (center + road_w//2, center - stripe_w//2 + offset, stripe_len, stripe_w))  # East


    # Directions: N=0, E=1, S=2, W=3
    car_start = [(center - 15, 100), (WIDTH - 100, center - 15), (center + 5, HEIGHT - 100), (100, center + 5)]
    car_deltas = [(0, 1), (-1, 0), (0, -1), (1, 0)]
    ped_start = [
        (center - 32, center - road_w//2 - crosswalk_len - 2),   # North (sidewalk)
        (center + road_w//2 + crosswalk_len + 2, center - 30),                   # East
        (center + 32, center + road_w//2 + crosswalk_len + 2),   # South
        (center - road_w//2 - crosswalk_len - 2, center + 30)    # West
    ]
    ped_deltas = [(0, 1), (-1, 0), (0, -1), (1, 0)]

    max_visible_cars = [3, 4, 3, 4]

    # Car queues
    for i in range(4):
        dx, dy = car_deltas[i]
        visible_queue_len = min(len(env.car_queues[i]), max_visible_cars[i])
        for j in range(visible_queue_len):
            x = car_start[i][0] + dx * (len(env.car_queues[i]) - 1 - j) * 24
            y = car_start[i][1] + dy * (len(env.car_queues[i]) - 1 - j) * 24

            if dx == 0:
                pygame.draw.rect(screen, CAR_COLOR, (x, y, 12, 20), border_radius=4)
            else:
                pygame.draw.rect(screen, CAR_COLOR, (x, y, 20, 12), border_radius=4)

    for i in range(4):
        dx, dy = car_deltas[i]
        for timer in env.crossing_cars_timers[i]:
            progress = (env.crossing_duration - timer + 1) / (env.crossing_duration + 1.5)

            # Travelling across the whole road to the far end crosswalk
            total_travel = 2 * (road_w + crosswalk_len + 20) + 200

            x = car_start[i][0] + dx * progress * total_travel
            y = car_start[i][1] + dy * progress * total_travel

            arrow_size = 6
            if dx == 0:
                arrow_tip = (int(x + 6), int(y - 5) if dy < 0 else int(y + 25))
                pygame.draw.polygon(screen, (255, 255, 255), [(arrow_tip[0], arrow_tip[1]),(arrow_tip[0] - 4, arrow_tip[1] - 6 * dy),(arrow_tip[0] + 4, arrow_tip[1] - 6 * dy)])
            else:
                arrow_tip = (int(x - 5) if dx < 0 else int(x + 25), int(y + 6))
                pygame.draw.polygon(screen, (255, 255, 255), [(arrow_tip[0], arrow_tip[1]),(arrow_tip[0] - 6 * dx, arrow_tip[1] - 4),(arrow_tip[0] - 6 * dx, arrow_tip[1] + 4)])

    # Pedestrian queues
    for i in range(4):
        dx, dy = ped_deltas[i]
        for j in range(len(env.ped_queues[i])):
            x = ped_start[i][0] + dx * j * 12
            y = ped_start[i][1] + dy * j * 12
            pygame.draw.circle(screen, PED_COLOR, (x, y), 4)

    # Crossing pedestrians
    for i in range(4):
        dx, dy = ped_deltas[i]
        for timer in env.crossing_peds_timers[i]:
            progress = (env.crossing_duration - timer + 1) / (env.crossing_duration + 1.5)
            x = ped_start[i][0] + dx * progress * (road_w + 60)
            y = ped_start[i][1] + dy * progress * (road_w + 60)
            pygame.draw.circle(screen, PED_COLOR, (int(x), int(y)), 4)

    pygame.display.flip()



def run():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Traffic Intersection")
    clock = pygame.time.Clock()

    env = TrafficIntersectionEnv()
    obs = env.reset()
    
    total_reward = 0  # <-- Initializing total reward

    running = True
    while running:
        clock.tick(2)  # Controlling the speed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = env.action_space.sample()  # Random agent
        obs, reward, done, _ = env.step(action)
        total_reward += reward  # <-- Accumulating reward here
        env.render()

        draw_env(screen, env)

    pygame.quit()
    
    print(f"Total reward accumulated by random agent: {total_reward}")  # <-- Final reward

if __name__ == "__main__":
    run()
