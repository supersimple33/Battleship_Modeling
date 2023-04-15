# BattleShip 2: Electric Boogaloo

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from shared import Space, setupShips

class Battleship3(gym.Env):
    """My third implementation of the battleship game."""
    metadata = {'render.modes': []} # ['human']}

    def __init__(self, render_mode = None): # do we want to set seeds here?
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # Space number to hit
        self.action_space = spaces.Discrete(100)
        # One hot encodes of misses, hits, and then binaries of sunk ships
        self.observation_space = spaces.Tuple((spaces.MultiBinary([10, 10, 6]), spaces.MultiBinary([5])))

        self.state = None
        self.hidState = None
    
    def reset(self, seed = None): # 42, runs ~ 10% faster than the original, ~
        super().reset(seed=seed)

        self.state = setupShips(self.np_random)
        self.hidState = np.full(shape=(2,10,10),fill_value=Space.Empty)
        
        self.counter = 0
        self.done = False
        self.reward = 0

        return self.hidState,

import timeit
env = Battleship3()
avg_list = []
for i in range(0,20):
    print(i)
    L = timeit.timeit('env.reset()', globals=globals(), number = 10000)
    # L = timeit.timeit(setup='env.reset()', stmt='for i in range(0,99): env.step(i); env.reset()', globals=globals(), number = 100) #2.73
    avg_list.append(L)
print("mean: ", sum(avg_list)/len(avg_list), "std_dev: ", np.std(avg_list))
print(avg_list)

# Trial set 1
# mean:  2.407954839600001 std_dev:  0.03430318664613518
# mean:  2.502668951999999 std_dev:  0.08318714081173365
# mean:  2.3938089833000005 std_dev:  0.015541127257026607