import gym
from gym import error, spaces, utils

import numpy as np

from shared import Space, setupShips

#game code
class Battleship1(gym.Env):
    metadata = {'render.modes': ['human']}
    seed = None

    def __init__(self, seed = None):
        self.seed = self.new_seed(seed)

        self.action_space = spaces.Discrete(100)
        missesChannel = spaces.Tuple([
            spaces.MultiDiscrete([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]) for _ in range(10)
        ])
        regChannel = spaces.Tuple([
            spaces.MultiDiscrete([3, 3, 3, 3, 3, 3, 3, 3, 3, 3]) for _ in range(10)
        ])
        self.observation_space = spaces.Tuple((missesChannel,regChannel,regChannel,regChannel,regChannel,regChannel))
        self.hidState = np.full(shape=(2,10,10),fill_value=Space.Empty)

        self.reset()

        # Action and observations spaces

    def _searchAndReplace(self, x, y, len_, search, replace, c):
        self.state[y][x] = replace
        self.hidState[c][y][x] = replace
        self.hidState[0][y][x] = Space.Empty
        direc = (-1,1)
        for d in direc:
            if (y+d) > 9 or (y+d) < 0:
                continue
            if len_ == 0:
                continue
            if self.state[y + d][x] == search:
                self.state[y + d][x] = replace
                self.hidState[c][y + d][x] = replace
                self.hidState[0][y + d][x] = Space.Empty
                len_ -= 1
                for eLow in range(1,len_):
                    e = eLow + 1
                    if self.state[y + (d * e)][x] == search:
                        self.state[y + (d * e)][x] = replace
                        self.hidState[c][y + (d * e)][x] = replace
                        self.hidState[0][y + (d * e)][x] = Space.Empty
                        len_ -= 1
                    else:
                        break
        for d in direc:
            if (x+d) > 9 or (x+d) < 0:
                continue
            if len_ == 0:
                continue
            if self.state[y][x + d] == search:
                self.state[y][x + d] = replace
                self.hidState[c][y][x + d] = replace
                self.hidState[0][y][x + d] = Space.Empty
                len_ -= 1
                for eLow in range(1,len_):
                    e = eLow + 1
                    if self.state[y][x + (d*e)] == search:
                        self.state[y][x + (d*e)] = replace
                        self.hidState[c][y][x + (d*e)] = replace
                        self.hidState[0][y][x + (d*e)] = Space.Empty
                        len_ -= 1
                    else:
                        break

    def step(self, target):
        x = target % 10
        y = target // 10
        targetSpace = self.state[y][x]
        self.reward = False
        # hit = False
        if self.done == True:
            # print("Game Over")
            return self.hidState, self.reward, self.done, self.expectedShots #check return
        else:
            if targetSpace == Space.Empty:
                self.state[y][x] = Space.Miss
                self.hidState[0][y][x] = Space.Miss
            elif targetSpace == Space.HiddenTwo:
                self.reward = True
                self.expectedShots[target] = Space.Empty
                self.state[y][x] = Space.HitPTwo
                self.hidState[0][y][x] = Space.HitPTwo
                self.hitsOnShips[4] = self.hitsOnShips[4] + 1
                if self.hitsOnShips[4] == 2:
                    self._searchAndReplace(x, y, self.hitsOnShips[4], Space.HitPTwo, Space.SunkTwo, 1)
            elif targetSpace == Space.HiddenSub:
                self.reward = True
                self.expectedShots[target] = Space.Empty
                self.state[y][x] = Space.HitPSub
                self.hidState[0][y][x] = Space.HitPSub
                self.hitsOnShips[3] = self.hitsOnShips[3] + 1
                if self.hitsOnShips[3] == 3:
                    self._searchAndReplace(x, y, self.hitsOnShips[3], Space.HitPSub, Space.SunkSub, 1)
            elif targetSpace == Space.HiddenCruiser:
                self.reward = True
                self.expectedShots[target] = Space.Empty
                self.state[y][x] = Space.HitPCruiser
                self.hidState[0][y][x] = Space.HitPCruiser
                self.hitsOnShips[2] = self.hitsOnShips[2] + 1
                if self.hitsOnShips[2] == 3:
                    self._searchAndReplace(x, y, self.hitsOnShips[2], Space.HitPCruiser, Space.SunkCruiser, 1)
            elif targetSpace == Space.HiddenFour:
                self.reward = True
                self.expectedShots[target] = Space.Empty
                self.state[y][x] = Space.HitPFour
                self.hidState[0][y][x] = Space.HitPFour
                self.hitsOnShips[1] = self.hitsOnShips[1] + 1
                if self.hitsOnShips[1] == 4:
                    self._searchAndReplace(x, y, self.hitsOnShips[1], Space.HitPFour, Space.SunkFour, 1)
            elif targetSpace == Space.HiddenFive:
                self.reward = True
                self.expectedShots[target] = Space.Empty
                self.state[y][x] = Space.HitPFive
                self.hidState[0][y][x] = Space.HitPFive
                self.hitsOnShips[0] = self.hitsOnShips[0] + 1
                if self.hitsOnShips[0] == 5:
                    self._searchAndReplace(x, y, self.hitsOnShips[0], Space.HitPFive, Space.SunkFive, 1)
            else:
                # print("Misfire")
                self.done = True
                self.reward = False
                return [self.hidState, self.reward, self.done, self.expectedShots]
            self.counter += 1
        win = self.hitsOnShips == [5,4,3,3,2]
        if win:
            self.done = True
            # print("Game over: ", self.counter, " moves.", sep = "", end = "\n")
            # self.reward = 100 - self.counter
        return self.hidState, self.reward, self.done, self.expectedShots

    def reset(self, seed=None):
        self.seed = self.new_seed(seed)

        self.state = setupShips(self.np_random)
        self.hidState.fill(Space.Empty)
        self.expectedShots = np.copy(np.reshape(self.state, (100)))

        self.hitsOnShips = [0, 0, 0, 0, 0]

        self.counter = 0
        self.done = False
        # self.add = [0, 0]
        self.reward = None
        return self.hidState, self.expectedShots
    
    def render(self, mode='human', close=False):
        ret = "0 "
        print("   0   1   2   3   4   5   6   7   8   9 ")
        i =  0
        for row in self.state:
            for slot in row:
                ret += slot.value[1] + " "
            print(ret)
            i += 1
            ret = str(i) + " "
        print()

    def new_seed(self, seed=None):
        self.np_random, seed = utils.seeding.np_random(seed)
        return seed

# full game 734 µs ± 27 µs per loop (mean ± std. dev. of 20 runs, 1000 loops each)

# 454 µs ± 20.4 µs per loop (mean ± std. dev. of 20 runs, 1000 loops each)

import timeit
env = Battleship1()
avg_list = []
for i in range(0,20):
    print(i)
    L = timeit.timeit('env.reset()', globals=globals(), number = 10000)
    # L = timeit.timeit(setup='env.reset()', stmt='for i in range(0,99): env.step(i); env.reset()', globals=globals(), number = 100) #2.73
    avg_list.append(L)
print("mean: ", sum(avg_list)/len(avg_list), "std_dev: ", np.std(avg_list))
print(avg_list)

# mean:  2.6814786437500002 std_dev:  0.028973936597862637
# mean:  2.6619473125999997 std_dev:  0.019689577780807894