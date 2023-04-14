# from gym.envs.registration import register # safety
from gymnasium.envs.registration import register

register(id='battleship1-v1', entry_point='gym_battleship1.envs:Battleship1') # safety here?
register(id='battleship3-v1', entry_point='gym_battleship2.envs:Battleship3') # watch entry