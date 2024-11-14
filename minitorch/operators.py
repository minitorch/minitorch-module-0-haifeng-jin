"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> bool:
    return x < y


def eq(x: float, y: float) -> bool:
    return x == y


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    if x > 100.0:
        return 1.0
    return 1.0 - 1.0 / (1.0 + math.exp(x))


def relu(x: float) -> float:
    return x if x > 0 else 0


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def inv(x: float) -> float:
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    return 1 / x * y


def inv_back(x: float, y: float) -> float:
    return -1.0 / (x * x) * y


def relu_back(x: float, y: float) -> float:
    return 0 if x < 0 else y


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

def map(fn, values):
    return [fn(value) for value in values]

def zipWith(fn, a, b):
    return [fn(a[i], b[i]) for i in range(min(len(a), len(b)))]

def reduce(fn, values, init_value):
    result = init_value
    for value in values:
        result = fn(result, value)
    return result

def addLists(a, b):
    return zipWith(add, a, b)

def negList(values):
    return map(neg, values)

def prod(values):
    return reduce(mul, values, 1)

def sum(values):
    return reduce(add, values, 0)

