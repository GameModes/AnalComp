#!/usr/bin/env python

"""Geneste objecten om simpele wiskundige formules te representeren, met als doel deze te differentiëren en integreren."""

__author__      = "Brian van der Bijl"
__copyright__   = "Copyright 2020, Hogeschool Utrecht"

from IPython.display import display, Math, Markdown
from collections import OrderedDict
from ac_exceptions import *
from ac_latex import latex_ratio
import math

def parentheses(str, outer):
    if outer:
        return str
    else:
        return "(" + str + ")"

class FormulaObject(object):
    def ask(self, question):
        return False

class Function(FormulaObject):
    def __init__(self, label, body, deriv_order=0):
        if isinstance(body, Function):
            raise SyntaxError("Nested functions not supported")
        self.body = body
        self.label = label
        self.deriv_order = deriv_order
    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.body == other.body
    def __str__(self):
        return "Function(label='" + self.label + "',body=" + str(self.body) + ",deriv_order=" + str(self.deriv_order) + ")"
    def simplify(self):
        return Function(self.label, self.body.simplify(), self.deriv_order)
    def complexity(self):
        return self.body.complexity()
    def variables(self):
        return [self.body.variables()]
    def deriv(self):
        return Function(self.label, self.body.deriv().simplify(), self.deriv_order + 1)
    def integrate(self, wrt):
        return Function(self.label, Sum(self.body.integrate(wrt).simplify(), Variable('C')), self.deriv_order - 1)
    def eval(self, vars):
        return self.body.eval(vars)
    def to_latex(self, jupyter_display=True, outer=True):
        if self.deriv_order >= 0:
            cont = self.label + "^{" + "\prime" * self.deriv_order + "}"
        else:
            cont = self.label.upper()
        if self.body.variables():
            cont += "(" + ','.join(list(OrderedDict.fromkeys(self.body.variables()))) + ")"
        cont += " = " + self.body.simplify().to_latex(outer=True)
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
        else:
            return cont

class Variable(FormulaObject):
    def __init__(self, label):
        self.label = label
    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.label == other.label
    def __str__(self):
        return "Variable(label='" + self.label + "')"
    def variables(self):
        if self.label != 'C':
            return [self.label]
        else:
            return []
    def deriv(self):
        return Constant(1)
    def simplify(self):
        return self
    def complexity(self):
        return 1
    def eval(self, vars):
        if self.label in vars:
            return vars[self.label]
        elif self.label == 'C': # C defaults to zero if not given
            return 0
        else:
            raise VariableError("Evaluating " + self.label + ", but no value provided")
        return self.body.eval(vars)
    def to_latex(self, jupyter_display=False, outer=False):
        cont = self.label
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
        else:
            return cont

class Power(FormulaObject):
    def __init__(self, base, exponent):
        self.base = base
        self.exponent = exponent
    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.base == other.base and self.exponent == other.exponent
    def __str__(self):
        return "Power(base=" + str(self.base) + ",exponent=" + str(self.exponent) + ")"
    def ask(self, question):
        if question == "negative_exponent":
            return self.exponent < 0
        elif question == "one":
            return self.exponent == 0
        else:
            return False
    def variables(self):
        return self.base.variables()
    def simplify(self):
        base = self.base.simplify()
        if self.exponent == 0:
            return Constant(1)
        elif self.exponent == 1:
            return self.base
        elif isinstance(base, Power):
            return Power(base.base, base.exponent * self.exponent)
        else:
            return self
    def complexity(self):
        return 1 + self.base.complexity()
    def eval(self, vars):
        return self.base.eval(vars) ** self.exponent
    def to_latex(self, jupyter_display=False, outer=False):
        if self.exponent == 0:
            cont = "1"
        elif self.exponent == 1:
            cont = self.base.to_latex()
        elif self.base.ask("trig_fn"):
            cont = r"\text{" + self.base.ask("trig_fn") + "}^{" + str(self.exponent) + "}" + self.base.argument.to_latex()
        elif self.exponent < 0:
            cont = r"\frac{1}{" + Power(self.base, abs(self.exponent)).to_latex() + "}"
        # ToDo: wortels ipv breuken afhankelijk van flag?
        else:
            cont = self.base.to_latex() + "^{" + latex_ratio(self.exponent) + "}" 
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
        else:
            return cont   

class Negative(FormulaObject):
    def __init__(self, inverse):
        self.inverse = inverse
    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.inverse == other.inverse
    def __str__(self):
        return "Negative(inverse=" + str(self.inverse) + ")"
    def ask(self, question):
        if question == "zero":
            return self.inverse == 0
        else:
            return False
    def variables(self):
        return self.inverse.variables()
    def deriv(self):
        return Negative(self.inverse.deriv())
    def integrate(self, wrt):
        return Negative(self.inverse.integrate(wrt))
    def simplify(self):
        if isinstance(self.inverse, Negative):
            return self.inverse.inverse
        elif isinstance(self.inverse.simplify(), Constant) and self.inverse.simplify().value == 0:
            return Constant(0)
        else:
            return Negative(self.inverse.simplify())
    def complexity(self):
        return 1 + self.inverse.complexity()
    def eval(self, vars):
        return - self.inverse.eval(vars)
    def to_latex(self, jupyter_display=False, outer=False):
        cont = "-" + self.inverse.to_latex()
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
        else:
            return cont

class Constant(FormulaObject):
    def __new__(cls, value):
        if value >= 0:
            return FormulaObject.__new__(cls)
        else:
            return Negative(Constant(abs(value)))
    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.value == other.value
    def __init__(self, value):
        if value >= 0:
            self.value = value
        else:
            raise Exception("Should not happen")
    def __str__(self):
        return "Constant(value=" + str(self.value) + ")"
    def ask(self, question):
        if question == "zero":
            return self.value == 0
        elif question == "one":
            return self.value == 1
        else:
            return False
    def variables(self):
        return []
    def deriv(self):
        return Constant(0)
    def simplify(self):
        return self
    def complexity(self):
        return 1
    def eval(self, vars):
        return self.value
    def to_latex(self, jupyter_display=False, outer=False):
        cont = latex_ratio(self.value)
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
        else:
            return cont

class Product(FormulaObject):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.left == other.left\
                                                                   and self.right == other.right
    def __str__(self):
        return "Product(left=" + str(self.left) + ",right=" + str(self.right) + ")"
    def variables(self):
        return self.left.variables() + self.right.variables()
    def simplify(self):
        left = self.left.simplify()
        right = self.right.simplify()
        if left.ask("zero") or right.ask("zero"):
            return Constant(0)
        elif left.ask("one"):
            return right.simplify()
        elif right.ask("one"):
            return left.simplify()
        elif isinstance(left, Constant) and isinstance(right, Constant):
            return Constant(left.value * right.value)
        elif isinstance(left, Negative):
            return Negative(Product(left.inverse, right))
        elif isinstance(right, Power) and isinstance(left, Power) and left.base == right.base:
            return Power(left.base, left.exponent + right.exponent)
        elif isinstance(right, Power) and isinstance(left, Power) and left.exponent == right.exponent:
            return Power(Product(left.base, right.base), left.exponent)
        elif isinstance(left, Power) and left.base == right:
            return Power(right, left.exponent + 1)
        elif isinstance(right, Power) and left == right.base:
            return Power(left, right.exponent + 1)
        elif isinstance(left, Product) and isinstance(right, Power) and left.right == right.base:
            return Product(left.left, Power(right.base, right.exponent + 1))
        elif left == right:
            return Power(left, 2)
        else:
            return Product(self.left.simplify(), self.right.simplify())
    def complexity(self):
        return 1 + max(self.left.complexity(), self.right.complexity())
    def eval(self, vars):
        return self.left.eval(vars) * self.right.eval(vars)
    def to_latex(self, jupyter_display=False, outer=False):
        if isinstance(self.left, Constant) and isinstance(self.right.simplify(), Variable):
            cont = self.left.to_latex() + self.right.to_latex()
        elif self.right.ask("negative_exponent"):
            cont = r"\frac{" + self.left.to_latex() + "}{" + Power(self.right.base, abs(self.right.exponent)).to_latex() + "}"
        else:
            #return parentheses((self.left.to_latex()) + " \  " + (self.right.to_latex()), outer)
            cont = (self.left.to_latex() + " \  " + self.right.to_latex())
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
        else:
            return cont

class Sum(FormulaObject):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.left == other.left\
                                                                   and self.right == other.right
    def __str__(self):
        return "Sum(left=" + str(self.left) + ",right=" + str(self.right) + ")"
    def variables(self):
        return self.left.variables() + self.right.variables()
    def simplify(self):
        left = self.left.simplify()
        right = self.right.simplify()
        if left.ask("zero"):
            return right.simplify()
        elif right.ask("zero"):
            return left.simplify()
        elif isinstance(left, Constant) and isinstance(right, Constant):
            return Constant(left.value * right.value)
        else:
            return Sum(self.left.simplify(), self.right.simplify())
    def complexity(self):
        return 1 + max(self.left.complexity(), self.right.complexity())
    def eval(self, vars):
        return self.left.eval(vars) + self.right.eval(vars)
    def to_latex(self, jupyter_display=False, outer=False):
        if isinstance(self.right, Negative):
            cont = parentheses((self.left.to_latex()) + " - " + (self.right.inverse.to_latex()), outer)
        else:
            cont = parentheses((self.left.to_latex()) + " + " + (self.right.to_latex()), outer)
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
        else:
            return cont

class Sin(FormulaObject):
    def __init__(self, argument):
        self.argument = argument
    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.argument == other.argument
    def __str__(self):
        return "Sin(argument=" + str(self.argument) + ")"
    def ask(self, question):
        if question == "trig_fn":
            return "sin"
        else:
            return False
    def variables(self):
        return self.argument.variables()
    def simplify(self):
        return self
    def complexity(self):
        return 1 + self.argument.complexity()
    def eval(self, vars):
        return math.sin(self.argument.eval(vars))
    def to_latex(self, jupyter_display=False, outer=False):
        cont = r"\text{sin}(" + self.argument.to_latex() + ")"
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
        else:
            return cont

class Cos(FormulaObject):
    def __init__(self, argument):
        self.argument = argument
    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.argument == other.argument
    def __str__(self):
        return "Cos(argument=" + str(self.argument) + ")"
    def ask(self, question):
        if question == "trig_fn":
            return "cos"
        else:
            return False
    def variables(self):
        return self.argument.variables()
    def simplify(self):
        return self
    def complexity(self):
        return 1 + self.argument.complexity()
    def eval(self, vars):
        return math.cos(self.argument.eval(vars))
    def to_latex(self, jupyter_display=False, outer=False):
        cont = r"\text{cos}(" + self.argument.to_latex() + ")"
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
        else:
            return cont

class Tan(FormulaObject):
    def __init__(self, argument):
        self.argument = argument
    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.argument == other.argument
    def __str__(self):
        return "Tan(argument=" + str(self.argument) + ")"
    def ask(self, question):
        if question == "trig_fn":
            return "tan"
        else:
            return False
    def variables(self):
        return self.argument.variables()
    def simplify(self):
        return self
    def complexity(self):
        return 1 + self.argument.complexity()
    def eval(self, vars):
        return math.tan(self.argument.eval(vars))
    def to_latex(self, jupyter_display=False, outer=False):
        cont = r"\text{tan}(" + self.argument.to_latex() + ")"
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
        else:
            return cont

class Cot(FormulaObject):
    def __init__(self, argument):
        self.argument = argument
    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.argument == other.argument
    def __str__(self):
        return "Cot(argument=" + str(self.argument) + ")"
    def ask(self, question):
        if question == "trig_fn":
            return "cot"
        else:
            return False
    def variables(self):
        return self.argument.variables()
    def simplify(self):
        return self
    def complexity(self):
        return 1 + self.argument.complexity()
    def deriv(self):
        return Negative(Product(self.argument.deriv(),
                                Power(Csc(Variable('x')), 2)))
    def eval(self, vars):
        return 1 / math.tan(self.argument.eval(vars))
    def to_latex(self, jupyter_display=False, outer=False):
        cont = r"\text{cot}(" + self.argument.to_latex() + ")"
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
        else:
            return cont

class Sec(FormulaObject):
    def __init__(self, argument):
        self.argument = argument
    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.argument == other.argument
    def __str__(self):
        return "Sec(argument=" + str(self.argument) + ")"
    def ask(self, question):
        if question == "trig_fn":
            return "sec"
        else:
            return False
    def variables(self):
        return self.argument.variables()
    def simplify(self):
        return self
    def complexity(self):
        return 1 + self.argument.complexity()
    def deriv(self):
        return Product(self.argument.deriv(),
                       Product(Sec(self.argument), Tan(self.argument)))
    def eval(self, vars):
        return 1 / math.cos(self.argument.eval(vars))
    def to_latex(self, jupyter_display=False, outer=False):
        cont = r"\text{sec}(" + self.argument.to_latex() + ")"
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
        else:
            return cont

class Csc(FormulaObject):
    def __init__(self, argument):
        self.argument = argument
    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.argument == other.argument
    def __str__(self):
        return "Csc(argument=" + str(self.argument) + ")"
    def ask(self, question):
        if question == "trig_fn":
            return "csc"
        else:
            return False
    def variables(self):
        return self.argument.variables()
    def simplify(self):
        return self
    def complexity(self):
        return 1 + self.argument.complexity()
    def deriv(self):
        return Negative(Product(self.argument.deriv(),
                                Product(Csc(self.argument), Cot(self.argument))))
    def eval(self, vars):
        return 1 / math.sin(self.argument.eval(vars))
    def to_latex(self, jupyter_display=False, outer=False):
        cont = r"\text{csc}(" + self.argument.to_latex() + ")"
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
        else:
            return cont  

class E(FormulaObject):
    def __init__(self, exponent):
        self.exponent = exponent
    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.exponent == other.exponent
    def __str__(self):
        return "E(exponent=" + str(self.exponent) + ")"
    def variables(self):
        return self.exponent.variables()
    def simplify(self):
        return self
    def complexity(self):
        return 1 + self.exponent.complexity()
    def eval(self, vars):
        return math.exp(self.exponent.eval(vars))
    def to_latex(self, jupyter_display=False, outer=False):
        cont = "e^{" + self.exponent.to_latex() + "}"
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
        else:
            return cont         

class Exponent(FormulaObject):
    def __init__(self, base, exponent):
        self.base = base
        self.exponent = exponent
    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.base == other.base\
                                                                   and self.exponent == other.exponent
    def __str__(self):
        return "Exponent(base=" + str(self.base) + ",exponent=" + str(self.exponent) + ")"
    def variables(self):
        return self.base.variables()+self.exponent.variables()
    def simplify(self):
        return self
    def complexity(self):
        return 1 + max(self.base.complexity(), self.exponent.complexity())
    def eval(self, vars):
        return self.base.eval(vars) ** self.exponent.eval(vars)
    def to_latex(self, jupyter_display=False, outer=False):
        cont = self.base.to_latex() + "^{" + self.exponent.to_latex() + "}"
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
        else:
            return cont

class Ln(FormulaObject):
    def __init__(self, argument):
        self.argument = argument
    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.argument == other.argument
    def __str__(self):
        return "Ln(argument=" + str(self.argument) + ")"
    def variables(self):
        return self.argument.variables()
    def simplify(self):
        return self
    def complexity(self):
        return 1 + self.argument.complexity()
    def eval(self, vars):
        return math.log(self.argument.eval(vars))
    def to_latex(self, jupyter_display=False, outer=False):
        cont = r"\text{ln}(" + self.argument.to_latex() + ")"
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
        else:
            return cont

class Log(FormulaObject):
    def __init__(self, base, argument):
        self.base = base
        self.argument = argument
    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.base == other.base\
                                                                   and self.argument == other.argument
    def __str__(self):
        return "Log(base=" + str(self.base) + ",exponent=" + str(self.argument) + ")"
    def variables(self):
        return self.argument.variables()
    def simplify(self):
        return self
    def complexity(self):
        return 1 + max(self.base.complexity(), self.argument.complexity())
    def eval(self, vars):
        return math.log(self.argument.eval(vars), self.base.eval(vars))
    def to_latex(self, jupyter_display=False, outer=False):
        cont = r"\text{log}_{" + self.base.to_latex() + "}(" + self.argument.to_latex() + ")"
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
        else:
            return cont  
