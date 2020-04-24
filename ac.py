#!/usr/bin/env python

"""Main import module voor de toets Analytical Computing."""

__author__      = "Brian van der Bijl"
__copyright__   = "Copyright 2020, Hogeschool Utrecht"

import math
import sys
import typing
import numpy as np
from IPython.display import display, Math, Markdown, YouTubeVideo, Code
from typing import Callable, Tuple, Dict, Union, List
import matplotlib.pyplot as plot

from ac_tests import *
from ac_random import *
from ac_latex import *
from ac_exceptions import *
from ac_formula import Function, Variable, Negative, Cot, Sec, Csc
import ac_formula

Polynomial = Tuple[Dict[int, float], str, str, int]

def polynomial(terms: Union[list, dict], label: str = 'f', var: str = 'x', primes: int = 0) -> Polynomial:
    if isinstance(terms, np.ndarray):
        terms = terms.flatten().tolist()
    if not isinstance(terms, dict):
        terms = dict(enumerate(terms))
    return (terms, label, var, primes)


def plot_data29(data, slope=None, intercept=None):
    xs = [ x for x in data] 
    ys = [ data[x] for x in xs ] 
    plot.scatter(xs, ys)
    plot.title("Number of people who died by becoming tangled in their bedsheets from per capita cheese consumption")
    plot.xlabel("cheese consumption")
    plot.ylabel("Deaths by bedsheet-tangling")
    if slope and intercept:
        axes = plot.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plot.plot(x_vals, y_vals, '--')
    plot.show()
