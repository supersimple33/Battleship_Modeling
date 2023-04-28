from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import restart, transition, termination
from tf_agents.environments import utils

import numpy as np
# from random import randint seeding replaces

from battleship_envs.envs.shared import Space, setup_ships, hidden_spaces, hit_spaces, sunk_spaces

HIT_SWAPS = {Space.HiddenFive: Space.HitPFive, Space.HiddenFour: Space.HitPFour, Space.HiddenCruiser: Space.HitPCruiser, Space.HiddenSub: Space.HitPSub, Space.HiddenTwo: Space.HitPTwo}
SUNK_SWAPS = {Space.HiddenFive: Space.SunkFive, Space.HiddenFour: Space.SunkFour, Space.HiddenCruiser: Space.SunkCruiser, Space.HiddenSub: Space.SunkSub, Space.HiddenTwo: Space.SunkTwo}
HIT_ORDERING = [Space.HiddenTwo, Space.HiddenSub, Space.HiddenCruiser, Space.HiddenFour, Space.HiddenFive]
SHIP_LENGTHS = [2, 3, 3, 4, 5]

#game code
class Battleship2(py_environment.PyEnvironment):
    metadata = {'render.modes': ['human']}

    def __init__(self, seed=None):
        self.np_random = np.random.default_rng(seed)

        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=99, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(2,10,10), dtype=np.float32, minimum=0, maximum=1, name='observation')
        # self._time_spec_step

        self._state = setup_ships(self.np_random) # unnecessary?

        self.hid_state = None
        self.dead_ships = None
        self.hits_on_ships = None
        self._episode_ended = False
        self._counter = None

    def action_spec(self): # Unused methods?
        return self._action_spec
    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = setup_ships(self.np_random)
        self.hid_state = np.full(shape=(2,10,10), fill_value=False, dtype=np.float32)
        self.expectedShots = np.reshape(self._state, (100))
        
        self.dead_ships = np.zeros(5, dtype=np.bool_)
        self.hits_on_ships = [0, 0, 0, 0, 0] # TODO: should we extract out dead_ships?

        self._counter = 0
        self._episode_ended = False
        # self.add = [0, 0]
        return restart(self.hid_state)

    def _search_and_replace(self, x: int, y: int, ship_len: int, search: Space, replace: Space):
        """Search for a certain space and replace it with another."""
        self._state[y][x] = replace
        ship_len -= 1
        direc = (-1, 1)
        # first check for replacements up and down
        for d in direc:
            # if the next step is out of bounds skip
            if (y+d) > 9 or (y+d) < 0:
                continue
            # check if a step in the search space
            if self._state[y+d][x] == search:
                self._state[y+d][x] = replace
                ship_len -= 1
                # iterate taking steps in the same direction
                for i in range(2, ship_len+2):
                    if (y+d*i) > 9 or (y+d*i) < 0:
                        break
                    if self._state[y+d*i][x] == search:
                        self._state[y+d*i][x] = replace
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
            if self._state[y][x+d] == search:
                self._state[y][x+d] = replace
                ship_len -= 1
                # iterate taking steps in the same direction
                for i in range(2, ship_len+2):
                    if (x+d*i) > 9 or (x+d*i) < 0:
                        break
                    if self._state[y][x+d*i] == search:
                        self._state[y][x+d*i] = replace
                        ship_len -= 1
                    else:
                        break
            if ship_len == 0:
                return
        raise RuntimeError("ship_len should be 0 at this point")

    def _step(self, target):
        """Take a step in the game shooting at the specified target."""
        # super().step(target)

        # VERIFICATIONS
        # assert target in self.action_space
        # assert self._verify()

        if self._episode_ended or self._counter >= 200:
            # raise ValueError("Game is over")
            return termination(self.hid_state, reward=0)

        self._counter += 1
        # self.done = self.counter >= 100 # do we want to turn off the game after 100 moves?
        reward = 0#-10

        x = target % 10
        y = target // 10

        # Did we hit an empty space?
        if self._state[y][x] == Space.Empty:
            self._state[y][x] = Space.Miss
            self.hid_state[0][y][x] = True
            reward = -1
        # Did we hit a ship?
        elif self._state[y][x] in hidden_spaces:
            slot = self._state[y][x]

            self._state[y][x] = HIT_SWAPS[slot]
            self.hid_state[1][y][x] = True
            reward = 1

            ship_num = HIT_ORDERING.index(slot)
            self.hits_on_ships[ship_num] += 1

            # does this shot sink a ship?
            if SHIP_LENGTHS[ship_num] == self.hits_on_ships[ship_num]: # speed up?
                self.dead_ships[ship_num] = True

                # self._state = fast_sinks[ship_num](self._state)
                # self._state[self._state == HIT_SWAPS[slot]] = SUNK_SWAPS[slot] # the easy way
                self._search_and_replace(x, y, SHIP_LENGTHS[ship_num], HIT_SWAPS[slot], SUNK_SWAPS[slot])
                
                # did we sink every ship?
                if self.dead_ships.all():
                    # print("Game Over")
                    self._episode_ended = True
        # Did we hit a ship we already sunk? uhoh
        elif self._state[y][x] in (hit_spaces | sunk_spaces | Space.Miss):
            reward = -10
        else:
            raise ValueError("Invalid state")
        # assumption no bad values
        # else:
        #     reward = -1000


        # assert (self.hid_state, self.dead_ships) in self.observation_space

        if self._episode_ended:
            return termination(self.hid_state, reward)
        else:
            return transition(self.hid_state, reward, discount=1.0)
    
    def _render(self, mode='human', close=False):
        ret = "0 "
        print("   0   1   2   3   4   5   6   7   8   9 ")
        i =  0
        for row in self._state:
            for slot in row:
                ret += slot.description() + " "
            print(ret)
            i += 1
            ret = str(i) + " "
        print()
