#!/usr/bin/env python

"""Diverse wiskundige structuren weergeven in LaTeX in Jupyter Notebook."""

__author__      = "Brian van der Bijl"
__copyright__   = "Copyright 2020, Hogeschool Utrecht"

from IPython.display import display, Math, Markdown
import re

def show_num(x):
    return re.compile(r"\.(?!\d)").sub("\1",x)

def latex_formula(form):
    latex = form.simplify().to_latex(outer=True)
    if latex:
        display(Math(latex))
        display(Markdown("<details><pre>$" + latex + "$</pre></details>"))

def latex_bmatrix(M, label=None): # Gebaseerd op https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix
    if len(M.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(M).replace("[", "").replace("]", "").splitlines()
    if label:
        result = [label + " = "]
    else:
        result = [""]
    result += [r"\begin{bmatrix}"]
    result += ["  " + " & ".join(map(show_num, l.split())) + r"\\" for l in lines]
    result +=  [r"\end{bmatrix}"]
    display(Math("\n".join(result)))
    display(Markdown("<details><pre>$" + " ".join(result) + "$</pre></details>"))

def latex_amatrix(M, labels=None):
    if len(M.shape) > 2:
        raise ValueError('array can at most display two dimensions')
    lines = str(M).replace("[", "").replace("]", "").splitlines()
    if labels and len(labels) == 2:
        result = [r"(\mathbf{" + labels[0] + r"} | \vec " + labels[1] + ") = "]
    else:
        result = [""]
    result += [r"\left[\begin{array}{ccc|c}"]
    result += ["  " + " & ".join(map(show_num, l.split())) + r"\\" for l in lines]
    result +=  [r"\end{array}\right]"]
    display(Math("\n".join(result)))
    display(Markdown("<details><pre>$" + " ".join(result) + "$</pre></details>"))

def latex_msquare(sq):
    if sq.shape != (3,3):
        raise ValueError('Geen magisch vierkant')
    lines = str(sq).replace("[", "").replace("]", "").splitlines()
    result = [r"\begin{array}{|c|c|c|}\hline"]
    result += ["  " + " & ".join(map(show_num, l.split())) + r"\\\hline" for l in lines]
    result +=  [r"\end{array}"]
    display(Math("\n".join(result)))
    display(Markdown("<details><pre>$" + " ".join(result) + "$</pre></details>"))

def latex_ratio(x):
    """Helper functie om breuken naar LaTeX te converteren; getallen worden alleen naar string
       geconverteerd."""
    if isinstance(x, int):
        return str(x)
    else:
        n, d = x.as_integer_ratio() # Nul buiten de breuk halen
        return ("-" if n < 0 else "") + r"\frac{" + str(abs(n)) + "}{" + str(d) + "}"

def latex_polynomial(poly):
    terms, label, var, primes = poly # Bind parameters uit tuple

    def power(exp):
        """Print een term (e.g. x^2). x^1 is gewoon x, x^0 is 1, maar n × 1 is gewoon n dus verberg de 1.
           In alle andere gevallen wordt de variabele met het juiste exponent opgeleverd."""
        if exp is 1:
            return var
        elif exp is 0:
            return ""
        else:
            return (var + r"^{" + latex_ratio(exp) + "}")

    # Print f(x) met het juiste aantal primes 
    result = label + ("^{" + r"\prime"*primes + "}" if primes > 0 else "") + "(" + var + ") = "
    first = True # Na de eerste moet er "+" tussen de termen komen

    for k, v in reversed(sorted(terms.items())): # Voor iedere term, van groot (hoog exponent) naar klein
        if v > 0 and not first: # Koppel met een plus, tenzij het de eerste term is
            result += "+"
        elif v < 0: # Koppel met een min als de term negatief is, ook de eerste term
            result += "-"

        if v != 0: # Zet first op False na de eerste keer
            first = False

        if k is 0:
            result += str(v)
        elif abs(v) is 1: # Print x in plaats van 1x en -x in plaats van -1x
            result += str(power(k))
        elif v != 0: # Print iedere term die niet 0 of 1 is op de gebruikelijke manier, zonder min want die staat
            result += latex_ratio(abs(v)) + str(power(k))  #   erboven al

    display(Math(result))
    display(Markdown("<details><pre>$" + result + "$</pre></details>"))
