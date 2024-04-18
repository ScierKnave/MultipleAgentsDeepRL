from functools import reduce
from operator import mul
from gym import spaces

def get_input_size(space):

    if type(space) == spaces.Box:
        return reduce(mul, space.shape)
    
    return space[0].n