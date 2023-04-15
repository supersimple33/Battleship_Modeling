import numpy as np
from enum import Enum, unique, IntEnum

# indList = ("|-|","!M!","(2)","(S)","(C)","(4)","(5)",3"x2x","xSx","xCx","x4x","x5x","|2|","|S|","|C|","|4|","|5|","HiddenCruiser")
@unique
class Space(Enum): #int enum for better performance?
    """The labels for each of the spaces in the battleship game."""
    Empty = np.float32(0.0),"|-|" #_

    Miss = np.float32(-1.0),"!M!" #m
    
    HitPTwo = np.float32(1.0),"(2)"
    HitPSub = np.float32(1.0),"(S)"
    HitPCruiser = np.float32(1.0),"(C)"
    HitPFour = np.float32(1.0),"(4)"
    HitPFive = np.float32(1.0),"(5)"

    SunkTwo = np.float32(0.2),"x2x" # ship values, should every ship get its own channel
    SunkSub = np.float32(0.4),"xSx"
    SunkCruiser = np.float32(0.6),"xCx"
    SunkFour = np.float32(0.8),"x4x"
    SunkFive = np.float32(1.0),"x5x"

    HiddenTwo = np.float32(1.0),"|2|" # Need to update these values likely 1
    HiddenSub = np.float32(1.0),"|S|"
    HiddenCruiser = np.float32(1.0),"|C|"
    HiddenFour = np.float32(1.0),"|4|"
    HiddenFive = np.float32(1.0),"|5|"

@unique
class Direction(IntEnum):
    """Directions for ships to be placed"""
    Up = 0
    Left = 1
    Down = 2
    Right = 3

# Spaces 2?

def addShip(state, ship: Space, ship_len: int, x: int, y: int, d: Direction) -> bool:
    """Given a state, a ship, a length, coords, and a direction, first check if this is a valid placement and then place the ship"""
    r = range(0, ship_len)
    if d == Direction.Up:# loop run twice in order to make sure all spaces are clear before making modifications
        for j in r:
            if state[y - j][x] != Space.Empty: 
                return False
        for j in r:
            state[y - j][x] = ship
    elif d == Direction.Left:
        for j in r:
            if state[y][x - j] != Space.Empty:
                return False
        for j in r:
            state[y][x - j] = ship
    elif d == Direction.Down:
        for j in r:
            if state[y + j][x] != Space.Empty:
                return False
        for j in r:
            state[y + j][x] = ship
    elif d == Direction.Right:
        for j in r:
            if state[y][x + j] != Space.Empty:
                return False
        for j in r:
            state[y][x + j] = ship
    return True

emptyStateRef = np.full(shape=(10,10),fill_value=Space.Empty)
hidSpaceRef = [Space.HiddenFive, Space.HiddenFour, Space.HiddenCruiser, Space.HiddenSub, Space.HiddenTwo]
shipSpaceLength = [5, 4, 3, 3, 2]
def setupShips(np_random: np.random.Generator): # need to make this very fast
    """Create a new state with ships placed randomly"""
    i = 0
    state = np.copy(emptyStateRef)
    # state.fill(Space.Empty) # redundant?
    slots = list(range(100))
    bad_slots = []
    while i < 5: #refactor out
        ship_len = shipSpaceLength[i]
        ship = hidSpaceRef[i]
        slot = np_random.choice(slots)
        x = slot % 10
        y = slot // 10
        d = np_random.choice( (Direction.Up, Direction.Down, Direction.Left, Direction.Right) )

        if (slot, d) in bad_slots: # don't redo work
            continue
        if not ((d == 0 and (y%10) - ship_len >= 0) or (d == 1 and (x%10) - ship_len >= 0) or (d == 2 and (y%10) + ship_len <= 9) or (d == 3 and (x%10) + ship_len <= 9)):
            bad_slots.append((slot, d))
            continue
        if (not addShip(state, ship, ship_len, x, y, d)): # could we add the ship, if not try again with new random coordinate
            bad_slots.append((slot, d))
            continue
        i += 1
    return state