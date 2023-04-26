# from gym.envs.registration import register # safety
# from gymnasium.envs.registration import register
from gymnasium import register

register(id='battleship1-v1', entry_point='battleship_envs.envs:Battleship1') # safety here?
register(id='battleship3-v1', entry_point='battleship_envs.envs:Battleship3') # watch entry
