import math
from ai.utils import Factory



def fn(x, /, y, *, log_base=math.e, exp_base=math.pi):
    return math.log(math.pow(exp_base, x) + math.pow(exp_base, y), log_base)


def test_factory():
    factory = Factory(fn, 2, 3, log_base=392, exp_base=7)
    assert(factory() == 1)


