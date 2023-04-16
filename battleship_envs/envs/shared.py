from enum import Flag, unique, IntEnum, auto
import numpy as np

# indList = ("|-|","!M!","(2)","(S)","(C)","(4)","(5)",3"x2x","xSx","xCx","x4x","x5x","|2|","|S|","|C|","|4|","|5|","HiddenCruiser")
@unique


class Space(Flag): # is this the best performance? Do we still need the numpy floats? intflag? bonus safety checks?
    """The labels for each of the spaces in the battleship game."""
    Empty = 0

    Miss = auto()

    HitPTwo = auto()
    HitPSub = auto()
    HitPCruiser = auto()
    HitPFour = auto()
    HitPFive = auto()

    SunkTwo = auto()
    SunkSub = auto()
    SunkCruiser = auto()
    SunkFour = auto()
    SunkFive = auto()

    HiddenTwo = auto()
    HiddenSub = auto()
    HiddenCruiser = auto()
    HiddenFour = auto()
    HiddenFive = auto()

    _ignore_ = {
        Empty : ("|-|", np.float32(0.0)),
        Miss : ("!M!", np.float32(-1.0)),
        HitPTwo : ("(2)", np.float32(1.0)),
        HitPSub : ("(S)", np.float32(1.0)),
        HitPCruiser : ("(C)", np.float32(1.0)),
        HitPFour : ("(4)", np.float32(1.0)),
        HitPFive : ("(5)", np.float32(1.0)),
        SunkTwo : ("x2x", np.float32(0.2)),
        SunkSub : ("xSx", np.float32(0.4)),
        SunkCruiser : ("xCx", np.float32(0.6)),
        SunkFour : ("x4x", np.float32(0.8)),
        SunkFive : ("x5x", np.float32(1.0)),
        HiddenTwo : ("|2|", np.float32(1.0)),
        HiddenSub : ("|S|", np.float32(1.0)),
        HiddenCruiser : ("|C|", np.float32(1.0)),
        HiddenFour : ("|4|", np.float32(1.0)),
        HiddenFive : ("|5|", np.float32(1.0)),
    }

    def description(self):
        return self._ignore_[self][0]
    
    def old_value(self):
        return self._ignore_[self][1]

hit_spaces = Space.HitPTwo | Space.HitPSub | Space.HitPCruiser | Space.HitPFour | Space.HitPFive
sunk_spaces = Space.SunkTwo | Space.SunkSub | Space.SunkCruiser | Space.SunkFour | Space.SunkFive
hidden_spaces = Space.HiddenTwo | Space.HiddenSub | Space.HiddenCruiser | Space.HiddenFour | Space.HiddenFive

@unique
class Direction(IntEnum):
    """Directions for ships to be placed"""
    Up = 0
    Left = 1
    Down = 2
    Right = 3

# Spaces 2?

def add_ship(state, ship: Space, ship_len: int, x: int, y: int, d: Direction) -> bool:
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
def setup_ships(np_random: np.random.Generator): # need to make this very fast
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
        if (not add_ship(state, ship, ship_len, x, y, d)): # could we add the ship, if not try again with new random coordinate
            bad_slots.append((slot, d))
            continue
        i += 1
    return state