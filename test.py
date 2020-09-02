from gym import spaces

discord = spaces.Tuple([
    spaces.MultiDiscrete([12, 12, 12, 12, 12, 12, 12, 12, 12, 12]) for _ in range(10)
])
