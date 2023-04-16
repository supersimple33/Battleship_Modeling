# BattleShip 2: Electric Boogaloo

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from shared import Space, setup_ships, hidden_spaces, hit_spaces, sunk_spaces

# CAPS
HIT_SWAPS = {Space.HiddenFive: Space.HitPFive, Space.HiddenFour: Space.HitPFour, Space.HiddenCruiser: Space.HitPCruiser, Space.HiddenSub: Space.HitPSub, Space.HiddenTwo: Space.HitPTwo}
SUNK_SWAPS = {Space.HiddenFive: Space.SunkFive, Space.HiddenFour: Space.SunkFour, Space.HiddenCruiser: Space.SunkCruiser, Space.HiddenSub: Space.SunkSub, Space.HiddenTwo: Space.SunkTwo}
HIT_ORDERING = [Space.HiddenTwo, Space.HiddenSub, Space.HiddenCruiser, Space.HiddenFour, Space.HiddenFive]

class Battleship3(gym.Env):
    """My third implementation of the battleship game."""
    metadata = {'render.modes': []} # ['human']}

    def __init__(self, render_mode = None): # do we want to set seeds here?
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # Space number to hit
        self.action_space = spaces.Discrete(100)
        # One hot encodes of misses, hits, and then binaries of sunk ships
        self.observation_space = spaces.Tuple((spaces.MultiBinary([10, 10, 2]), spaces.MultiBinary([5])))

        self.state = None
        self.hid_state = None
        self.reward_range = (float("-inf"), 17) # i think these numbers are right, may need tweaking
    
    def reset(self, seed = None): # 42, runs ~ 10% faster than the original, ~
        """Reset the game to an initial state and return a blank observation."""
        super().reset(seed=seed)

        self.state = setup_ships(self.np_random)
        self.hid_state = np.full(shape=(2,10,10), fill_value=False)
        self.dead_ships = np.zeros(5, dtype=np.bool_)
        
        self.counter = 0
        self.done = False

        return (self.hid_state, self.dead_ships), {}
    
    def step(self, target): 
        """Take a step in the game shooting at the specified target."""
        # super().step(target)
        assert target in self.action_space

        if self.done:
            # raise ValueError("Game is over")
            return (self.hid_state, self.dead_ships), 0, self.done, {}

        self.counter += 1
        # self.done = self.counter >= 100 # do we want to turn off the game after 100 moves?
        reward = 0

        x = target % 10
        y = target // 10

        # Did we hit an empty space?
        if self.state[y][x] == Space.Empty:
            self.state[y][x] = Space.Miss
            self.hid_state[0][y][x] = True
            reward = -1
        # Did we hit a ship?
        elif self.state[y][x] in hidden_spaces:
            slot = self.state[y][x]

            self.state[y][x] = HIT_SWAPS[slot]
            self.hid_state[1][y][x] = True
            reward = 1

            # does this shot sink a ship?
            if slot not in self.state:
                self.dead_ships[HIT_ORDERING.index(slot)] = True
                self.state[self.state == slot] = SUNK_SWAPS[slot]
                # did we sink every ship?
                if self.dead_ships.all():
                    # print("Game Over")
                    self.done = True
        # Did we hit a ship we already sunk? uhoh
        elif self.state[y][x] in hit_spaces or self.state[y][x] in sunk_spaces:
            reward = -10
        else:
            raise ValueError("Invalid state")

        return (self.hid_state, self.dead_ships), reward, self.done, {}


import timeit
env = Battleship3()
avg_list = []
for i in range(0,20):
    print(i)
    # L = [timeit.timeit('env.reset()', globals=globals(), number = 10000)]
    L = timeit.repeat(setup='env.reset(); i=0', stmt='env.step(i); i += 1', globals=globals(), number = 100, repeat = 5000) #2.73
    avg_list.append(sum(L))
print("mean: ", sum(avg_list)/len(avg_list), "std_dev: ", np.std(avg_list))
print(avg_list)

# Trial set 1
# mean:  2.407954839600001 std_dev:  0.03430318664613518
# mean:  2.502668951999999 std_dev:  0.08318714081173365
# mean:  2.3938089833000005 std_dev:  0.015541127257026607

# mean:  2.4946897120000466 std_dev:  0.004272984706538216
