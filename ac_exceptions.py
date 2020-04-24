#!/usr/bin/env python

"""Errors in een aparte module om import-cycli te vermijden."""

__author__      = "Brian van der Bijl"
__copyright__   = "Copyright 2020, Hogeschool Utrecht"

class DimensionError(Exception):
    pass

class NonInvertibleError(Exception):
    pass

class SyntaxError(Exception):
    pass

class VariableError(Exception):
    pass