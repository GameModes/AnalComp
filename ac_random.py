#!/usr/bin/env python

"""Generen van random matrices, vectoren, etc. voor studentenopgaven lineaire algebra en calculus."""

__author__      = "Brian van der Bijl"
__copyright__   = "Copyright 2020, Hogeschool Utrecht"

import numpy as np
from IPython.display import display, Math, Markdown
from ac_latex import *

class RNG:
    class __RNG:
        def __init__(self, studentnr):
            self.seed = studentnr
            np.random.seed(self.seed)

        def set(self, offset):
            s = np.random.get_state()[1][0]
            np.random.seed(self.seed + offset)
            return self

        def consume_entropy(self, n, a, b):
            np.random.randint(a, b, size=n)
            return self

    instance = None

    def __init__(self, studentnr=None):
        if not RNG.instance:
            try:
                RNG.instance = RNG.__RNG(int(studentnr))
                display(Markdown("<h3 style=\"color:#00cccc;\">Seed geïnitialiseerd.</h3>"))

            except ValueError:
                display(Markdown("<h2 style=\"color:red;\">Je bent vergeten je studentnummer in te vullen, doe dat op de eerste regel!</h2>"))
                raise ValueError("Je bent vergeten je studentnummer in te vullen, doe dat op de eerste regel!") from None

        elif studentnr:
                display(Markdown("<h2 style=\"color:orange;\">Seed reeds geïnitialiseerd.</h2>"))
                return None
        else:
            return None

    def __getattr__(self, name):
        return getattr(self.instance, name)

matrix_gd =  1
matrix_ns =  0
matrix_nd = -1

def random_tensor(label=None, size=None, singular=0, interval=None):
    def generate_tensor(size, interval):
        if not interval:
            interval = (-20, 20)
        if size and isinstance(size, int):
            size=(size,1)
        if size and isinstance(size, tuple):
            size=size
        else:
            size=(np.random.randint(2,6),1)
        return np.random.randint(interval[0], interval[1], size=size)
    candidate = generate_tensor(size, interval)
    while (singular ==  1 and np.linalg.det(candidate) != 0) or (singular == -1 and np.linalg.det(candidate) == 0):
            candidate = generate_tensor(size, interval)
    latex_bmatrix(candidate, label)

def random_scalar(label=None):
    if label:
        label = label
    else:
        label = ""
    display(Math(label + " = " + str(np.random.randint(-10,10))))

def random_matrix_vector(label=None,size=None):
    if size:
        size=(size,1)
    else:
        size=(np.random.randint(2,6),1)
    latex_bmatrix(np.random.randint(-20, 20, size=size), label)

def random_sys_of_eq():
    y = np.random.choice(9,3, False)
    Mi = np.random.choice(3,(3,3))
    while np.linalg.det(Mi) == 0:
        Mi = np.random.choice(3,(3,3))
    latex_amatrix(np.concatenate((Mi, np.reshape(np.linalg.det(Mi)*y, (3,1))), 1).astype(int), ("A", "b"))

def random_derivatives():
    def ho(x):
        if x == 1:
            return ""
        else:
            return x

    a,b,c = np.random.randint(2,7,3)
    text = f"Gegeven $f(x) = ({a-1}- {ho(b)}x)^{ho(c)}$, bepaal $f^\\prime(x)$"
    display(Markdown("**(a)** " + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))

    a,b,c,d = np.random.randint(2,7,4)
    text = f"Gegeven $g(x) = {ho(a-1)}x^{ho(b)}\\ \\text{{tan}}({ho(c)}x^{ho(d)})$, geef $g^\\prime(x)$"
    display(Markdown("**(b)** " + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))

    a,b,c,d = np.random.randint(2,7,4)
    text = f"Gegeven $h(x) = \\text{{log}}_{a}({b-1}x-{c}x^{d})$, geef $h^\\prime(x)$"
    display(Markdown("**(c)** " + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))

    a,b = np.random.randint(2,7,2)
    text = f"Gegeven $k(x) = \\frac{{{a}}}{{x^{b}}}$, geef $k^{{\\prime\\prime}}(x)$"
    display(Markdown("**(d)** " + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))

    a,b = np.random.randint(2,7,2)
    text = f"Gegeven $\\frac{{dy}}{{dx}} = x^{a} - {ho(b-1)}y$, geef $\\frac{{d^2y}}{{dx^2}}$"
    display(Markdown("**(e)** " + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))

    a,b,c,d = np.random.randint(2,7,4)
    text = f"Gegeven ${ho(a-1)}x^3y - {ho(b-1)}x^2 + {ho(c-1)}y^4 = {2*d}$, geef $\\frac{{dy}}{{dx}}$"
    display(Markdown("**(f)** " + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))

def random_integrals():
    def ho(x):
        if x == 1:
            return ""
        else:
            return x

    def frac(x):
        if x % 2 is 0:
            return str(x/2)
        else:
            return r"\frac{" + str(x) + "}{2}"

    a,b = np.random.randint(2,7,2)
    text = f"$$\\int \\sqrt[{a}]x^{b}\\ dx$$"
    display(Markdown("**(a)** Bereken " + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))

    a,b,c = np.random.randint(2,7,3)
    text = f"$$\\int_{min(a,b)}^{max(a,b)+2} {c}e^x\\ dx$$"
    display(Markdown("**(b)** Bereken " + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))

    a,b,c = np.random.randint(2,5,3)
    text = f"$$\int_{{{frac(min(a,b))}\\pi}}^{{{frac(max(a,b)+2)}\\pi}} -{c} \\text{{sin}}(x)\\ dx$$"
    display(Markdown("**(c)** Bereken " + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))

    a,b,c,d = np.random.randint(3,9,4)
    text = f"$$\\int ({a*b}x^{b-1})({a}x^{b}+{c})^{d}\\ dx$$"
    display(Markdown("**(d)** Bereken " + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))

    a,b,c = np.random.randint(2,7,3)
    text = f"$$\\int ({a}x^{b})\\text{{log}}_{c}(x)\\ dx$$"
    display(Markdown("**(e)** Bereken " + text))
    display(Markdown("<details><pre>" + text + "</pre></details>"))

def random_de():
    a,b,c,d,e,f,g,h = np.random.randint(2,9,8)
    b = int(b/2)
    c = c*d
    d = 3*e*f
    e = 2*(g-4)
    f = h*2 -  1
    deriv = f"f^\\prime(x) = {a*b}x^{b-1}+{c}e^x"
    val = f"f({e}) = {a*(e**b)-d}+{c}e^{{{e}}}"
    display(Markdown(f"Vind $f({f})$ gegeven de volgende afgeleidde en waarde:\n\n$${deriv},\ {val}$$"))
    display(Markdown("LaTeX-code van de afgeleide: <details><pre>" + deriv + "</pre></details>"))
    display(Markdown(f"LaTeX-code van de functie op {e}: <details><pre>" + val + "</pre></details>"))
