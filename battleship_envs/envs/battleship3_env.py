# BattleShip 2: Electric Boogaloo

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from battleship_envs.envs.shared import Space, setup_ships, hidden_spaces, hit_spaces, sunk_spaces, check_hittable, fast_sinks

# CAPS
HIT_SWAPS = {Space.HiddenFive: Space.HitPFive, Space.HiddenFour: Space.HitPFour, Space.HiddenCruiser: Space.HitPCruiser, Space.HiddenSub: Space.HitPSub, Space.HiddenTwo: Space.HitPTwo}
SUNK_SWAPS = {Space.HiddenFive: Space.SunkFive, Space.HiddenFour: Space.SunkFour, Space.HiddenCruiser: Space.SunkCruiser, Space.HiddenSub: Space.SunkSub, Space.HiddenTwo: Space.SunkTwo}
HIT_ORDERING = [Space.HiddenTwo, Space.HiddenSub, Space.HiddenCruiser, Space.HiddenFour, Space.HiddenFive]
SHIP_LENGTHS = [2, 3, 3, 4, 5]

class Battleship3(gym.Env):
    """My third implementation of the battleship game."""
    metadata = {'render.modes': []} # ['human']}

    def __init__(self, render_mode = None): # do we want to set seeds here?
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # Space number to hit
        self.action_space = spaces.Discrete(100)
        # One hot encodes of misses, hits, and then binaries of sunk ships
        self.observation_space = spaces.Tuple((spaces.MultiBinary([2, 10, 10]), spaces.MultiBinary([5])))

        self.state = None
        self.hid_state = None
        self.reward_range = (float("-inf"), 17) # i think these numbers are right, may need tweaking
    
    def reset(self, seed = None): # 42, runs ~ 10% faster than the original, ~
        """Reset the game to an initial state and return a blank observation."""
        super().reset(seed=seed)

        self.state = setup_ships(self.np_random)
        self.hid_state = np.full(shape=(2,10,10), fill_value=False, dtype=np.bool_)
        self.dead_ships = np.zeros(5, dtype=np.bool_)
        self.hits_on_ships = [0, 0, 0, 0, 0] # TODO: should we extract out dead_ships?
        
        self.counter = 0
        self.done = False

        # assert (self.hid_state, self.dead_ships) in self.observation_space

        return (self.hid_state, self.dead_ships), {}

    def _search_and_replace(self, x: int, y: int, ship_len: int, search: Space, replace: Space):
        """Search for a certain space and replace it with another."""
        self.state[y][x] = replace
        ship_len -= 1
        direc = (-1, 1)
        # first check for replacements up and down
        for d in direc:
            # if the next step is out of bounds skip
            if (y+d) > 9 or (y+d) < 0:
                continue
            # check if a step in the search space
            if self.state[y+d][x] == search:
                self.state[y+d][x] = replace
                ship_len -= 1
                # iterate taking steps in the same direction
                for i in range(2, ship_len+2):
                    if (y+d*i) > 9 or (y+d*i) < 0:
                        break
                    if self.state[y+d*i][x] == search:
                        self.state[y+d*i][x] = replace
                        ship_len -= 1
                    else:
                        break # just break the inners
            if ship_len == 0:
                return
        # then check for replacements left and right
        for d in direc:
            # if the next step is out of bounds skip
            if (x+d) > 9 or (x+d) < 0:
                continue
            # check if taking a step gives a hit
            if self.state[y][x+d] == search:
                self.state[y][x+d] = replace
                ship_len -= 1
                # iterate taking steps in the same direction
                for i in range(2, ship_len+2):
                    if (x+d*i) > 9 or (x+d*i) < 0:
                        break
                    if self.state[y][x+d*i] == search:
                        self.state[y][x+d*i] = replace
                        ship_len -= 1
                    else:
                        break
            if ship_len == 0:
                return
        raise RuntimeError("ship_len should be 0 at this point")

    
    def step(self, target): # 
        """Take a step in the game shooting at the specified target."""
        # super().step(target)

        # VERIFICATIONS
        # assert target in self.action_space
        # assert self._verify()

        if self.done:
            # raise ValueError("Game is over")
            return (self.hid_state, self.dead_ships), 0, self.done, {}

        self.counter += 1
        # self.done = self.counter >= 100 # do we want to turn off the game after 100 moves?
        reward = 0#-10

        x = target % 10
        y = target // 10

        # Did we hit an empty space?
        if self.state[y][x] == Space.Empty:
            self.state[y][x] = Space.Miss
            self.hid_state[0][y][x] = True
        # Did we hit a ship?
        elif self.state[y][x] in hidden_spaces:
            slot = self.state[y][x]

            self.state[y][x] = HIT_SWAPS[slot]
            self.hid_state[1][y][x] = True
            # reward += 20

            ship_num = HIT_ORDERING.index(slot)
            self.hits_on_ships[ship_num] += 1

            # does this shot sink a ship?
            if SHIP_LENGTHS[ship_num] == self.hits_on_ships[ship_num]: # speed up?
                self.dead_ships[ship_num] = True

                # self.state = fast_sinks[ship_num](self.state)
                # self.state[self.state == HIT_SWAPS[slot]] = SUNK_SWAPS[slot] # the easy way
                self._search_and_replace(x, y, SHIP_LENGTHS[ship_num], HIT_SWAPS[slot], SUNK_SWAPS[slot])
                
                # did we sink every ship?
                if self.dead_ships.all():
                    # print("Game Over")
                    self.done = True
        # Did we hit a ship we already sunk? uhoh
        elif self.state[y][x] in (hit_spaces | sunk_spaces | Space.Miss):
            reward = -100
        else:
            raise ValueError("Invalid state")
        # assumption no bad values
        # else:
        #     reward = -1000


        # assert (self.hid_state, self.dead_ships) in self.observation_space

        return (self.hid_state, self.dead_ships), reward, self.done, {}
    
    def _verify(self) -> bool:
        """Runs some light verifications that the current state is one that actually makes sense."""
        for i in range(5):
            # if a ship has been hit n times ensure there are n hits on the board
            if self.hits_on_ships[i] < SHIP_LENGTHS[i]:
                if self.hits_on_ships[i] != np.count_nonzero(self.state == (HIT_SWAPS[HIT_ORDERING[i]])):
                    print(f"Hits on ship {i} is wrong!")
                    return False

            # dead ship checks
            if self.dead_ships[i]:
                # if a ship is sunk ensure there are x sunk spaces on the board
                if np.count_nonzero(self.state == SUNK_SWAPS[HIT_ORDERING[i]]) != SHIP_LENGTHS[i]:
                    print(f"Ship {i} is sunk but has the wrong number of sunk spaces!")
                    return False

                # if a ship is sunk ensure there are no hits or hiddens still on the board
                if np.count_nonzero(self.state == HIT_SWAPS[HIT_ORDERING[i]]) + np.count_nonzero(self.state == HIT_ORDERING[i]) != 0:
                    print(f"Ship {i} is sunk but has hits on it!")
                    return False

                # check that if a ship is sunk, it has the right number of hits
                if self.hits_on_ships[i] != SHIP_LENGTHS[i]:
                    print(f"Ship {i} is sunk but has the wrong number of hits!")
                    return False
        if self.done:
            # if the game is over ensure all ships are sunk
            if not self.dead_ships.all():
                print("Game is over but not all ships are sunk!")
                return False
        else:
            # if the game is not over ensure not all ships are sunk
            if self.dead_ships.all():
                print("Game is not over but all ships are sunk!")
                return False

            # ensure there are at least still some slots left to hit
            if not np.any(check_hittable(self.state)):
                print("Game is not over but there are no spaces left!")
                return False
        if np.any(np.logical_and(self.hid_state[0], self.hid_state[1])):
            print("Hidden state has a space that is both hit and missed!")
            return False
        return True


# import timeit
# env = Battleship3()
# env.reset()
# env._verify()
# avg_list = []
# for i in range(0,20):
#     print(i)
#     # L = [timeit.timeit('env.reset()', globals=globals(), number = 10000)]
#     L = timeit.repeat(setup='env.reset(); i=0', stmt='env.step(i); i += 1', globals=globals(), number = 100, repeat = 5000)
#     avg_list.append(sum(L))
# print("mean: ", sum(avg_list)/len(avg_list), "std_dev: ", np.std(avg_list))
# print(avg_list)

# Trial set 1
# mean:  2.407954839600001 std_dev:  0.03430318664613518
# mean:  2.502668951999999 std_dev:  0.08318714081173365
# mean:  2.3938089833000005 std_dev:  0.015541127257026607

# mean:  2.4946897120000466 std_dev:  0.004272984706538216
# mean:  2.482905437600021 std_dev:  0.00925681537253497

# mean:  1.9700759035000484 std_dev:  0.012570561470121295

# mean:  1.8528809404499484 std_dev:  0.0038643436997037235

# mean:  1.8864627204000057 std_dev:  0.05564629454365078
# mean:  1.868333858099994 std_dev:  0.04724201701465127