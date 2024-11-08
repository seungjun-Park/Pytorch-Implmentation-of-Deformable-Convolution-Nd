from collections import abc

from itertools import repeat
from typing import Union, List, Tuple

def _ntuple(n):
    def parse(x) -> Tuple:
        if isinstance(x, abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def multiply_integers(x) -> int:
    mul = 1
    if not isinstance(x, abc.Iterable):
        return x
    for i in x:
        mul *= i

    return int(mul)

