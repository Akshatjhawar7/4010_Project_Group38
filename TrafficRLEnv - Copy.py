import gym
from gym import spaces
import numpy as np
from gym.envs.registration import register

class TrafficIntersectionEnv(gym.Env):
    def __init__(self):
        super(TrafficIntersectionEnv, self).__init__()

        # Configuring
        self.num_lanes = 4  # N, E, S, W
        self.max_queue = 10
        self.yellow_duration = 3
        self.min_green_duration = 5

        # Encoding phases
        self.GREEN_NS = 0
        self.YELLOW_NS = 1
        self.GREEN_EW = 2
        self.YELLOW_EW = 3
        self.crossing_duration = 6

        # Defining action and observation space
        self.action_space = spaces.Discrete(2)  # 0 = hold, 1 = request switch
        self.observation_space = spaces.Box(
            low=0,
            high=self.max_queue,
            shape=(18,),
            dtype=np.int32
        )

        self.reset()

    def reset(self):
        self.car_queues = [[] for _ in range(4)]  # N, E, S, W
        self.ped_queues = [[] for _ in range(4)]
        self.crossing_cars_timers = [[] for _ in range(4)]
        self.crossing_peds_timers = [[] for _ in range(4)]
        self.arriving_cars_timers = [[] for _ in range(4)]

        self.phase = self.GREEN_NS
        self.phase_timer = 0
        self.total_steps = 0

        return self._get_obs()

    def _get_obs(self):
        car_queue_counts = np.array([len(q) for q in self.car_queues], dtype=np.int32)
        ped_queue_counts = np.array([len(q) for q in self.ped_queues], dtype=np.int32)
        crossing_cars = np.array([len(t) for t in self.crossing_cars_timers], dtype=np.int32)
        crossing_peds = np.array([len(t) for t in self.crossing_peds_timers], dtype=np.int32)
        return np.concatenate([
            car_queue_counts,
            ped_queue_counts,
            crossing_cars,
            crossing_peds,
            [self.phase],
            [self.phase_timer],
        ])

    def step(self, action):
        reward = 0
        done = False

        # Arrival simulation (simple Bernoulli)
        for i in range(4):
            if np.random.rand() < 0.2 and len(self.car_queues[i]) < self.max_queue:
                self.car_queues[i].append("car")
                self.arriving_cars_timers[i].append(4)
            if np.random.rand() < 0.05 and len(self.ped_queues[i]) < self.max_queue:
                self.ped_queues[i].append("ped")

        # Computing waiting penalty
        reward -= sum(len(q) for q in self.car_queues)
        reward -= sum(len(q) for q in self.ped_queues)

        switching = False

        # Handling switching
        if action == 1:
            if self.phase in [self.GREEN_NS, self.GREEN_EW] and self.phase_timer >= self.min_green_duration:
                self.phase = self.YELLOW_NS if self.phase == self.GREEN_NS else self.YELLOW_EW
                self.phase_timer = 0
                switching = True
            else:
                reward -= 10  # Penalty for switching too early

        elif self.phase in [self.YELLOW_NS, self.YELLOW_EW]:
            if self.phase_timer >= self.yellow_duration:
                self.phase = self.GREEN_EW if self.phase == self.YELLOW_NS else self.GREEN_NS
                self.phase_timer = 0

        # Movement logic
        if self.phase == self.GREEN_NS:
            for i in [0, 2]:  # N, S
                if self.car_queues[i]:
                    self.car_queues[i].pop(0)
                    self.crossing_cars_timers[i].append(self.crossing_duration)
                if self.ped_queues[i]:
                    self.ped_queues[i].pop(0)
                    self.crossing_peds_timers[i].append(self.crossing_duration)

        elif self.phase == self.GREEN_EW:
            for i in [1, 3]:  # E, W
                if self.car_queues[i]:
                    self.car_queues[i].pop(0)
                    self.crossing_cars_timers[i].append(self.crossing_duration)
                if self.ped_queues[i]:
                    self.ped_queues[i].pop(0)
                    self.crossing_peds_timers[i].append(self.crossing_duration)

        if switching:
            reward -= 5  # Switching cost

        # Updating crossing timers
        for i in range(4):
            self.crossing_cars_timers[i] = [t - 1 for t in self.crossing_cars_timers[i] if t > 1]
            self.crossing_peds_timers[i] = [t - 1 for t in self.crossing_peds_timers[i] if t > 1]
            self.arriving_cars_timers[i] = [t - 1 for t in self.arriving_cars_timers[i] if t > 1]

        self.phase_timer += 1
        self.total_steps += 1

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.total_steps}")
        print(f"Phase: {self.phase} (Timer: {self.phase_timer})")
        print(f"Car Queues: {[len(q) for q in self.car_queues]}")
        print(f"Ped Queues: {[len(q) for q in self.ped_queues]}")
        print(f"Crossing Cars: {[len(t) for t in self.crossing_cars_timers]}")
        print(f"Crossing Peds: {[len(t) for t in self.crossing_peds_timers]}")
        print("-" * 40)


# Registering the environment
register(
    id='TrafficIntersection-v0',
    entry_point='TrafficRLEnv:TrafficIntersectionEnv',
)
