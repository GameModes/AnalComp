#!/usr/bin/env python

"""Testsuite voor de toets Analytical Computing."""

__author__      = "Brian van der Bijl"
__copyright__   = "Copyright 2020, Hogeschool Utrecht"

import unittest as tst
import numpy as np
from ac_exceptions import *
from ac_formula import *

suite = tst.TestSuite()

def run_tests(test):
    try:
        shell = get_ipython()
        ipython = True
    except NameError:
        ipython = False

    tl = tst.TestLoader().loadTestsFromTestCase(test)

    if ipython:
        s = tst.TestSuite()
        s.addTest(tl)
        tst.TextTestRunner(verbosity=2).run(s)
    else:
        suite.addTest(tl)

def test_vector_addition(vector_addition):
    class TestVectorAddition(tst.TestCase):
        u = np.array((1, 2, 3))
        v = np.array((4, 5, 6))
        w = np.array((7,8))

        def test_valid_addition(self):
            np.testing.assert_array_equal(vector_addition(self.u, self.v), np.array((5,7,9)))
        def test_additive_unit(self):
            np.testing.assert_array_equal(vector_addition(self.v, np.zeros(3)), self.v)
        def test_additive_unit_2(self):
            np.testing.assert_array_equal(vector_addition(np.zeros(3), self.v), self.v)
        def test_invalid_addition(self):
            with self.assertRaises(DimensionError):
                vector_addition(self.v, self.w)
    run_tests(TestVectorAddition)

def test_negative_of_vector(negative_of_vector, vector_addition):
    class TestNegativeOfVector(tst.TestCase):
        v = np.array((1, 2, 3))
        z = np.zeros(3)

        def test_negative_of_vector(self):
            np.testing.assert_array_equal(negative_of_vector(self.v), -self.v)
        def test_negative_of_zero(self):
            np.testing.assert_array_equal(negative_of_vector(self.z), self.z)
        def test_sum_vector_negative_is_zero(self):
            np.testing.assert_array_equal(vector_addition(self.v, negative_of_vector(self.v)), self.z)
    run_tests(TestNegativeOfVector)

def test_scalar_product(scalar_product, vector_addition):
    class TestScalarProduct(tst.TestCase):
        a = 4
        b = 9
        u = np.array((1, 2, 3))
        v = np.array((5, 6, 7))
        z = np.zeros(3)

        def test_zero(self):
            np.testing.assert_array_equal(scalar_product(0, self.v), self.z)
        def test_unit(self):
            np.testing.assert_array_equal(scalar_product(1, self.v), self.v)
        def test_double(self):
            np.testing.assert_array_equal(scalar_product(2, self.v), vector_addition(self.v, self.v))
        def test_distributive_vector(self):
            np.testing.assert_array_equal(scalar_product(self.a, vector_addition(self.u, self.v)), 
                                          vector_addition(scalar_product(self.a, self.u), 
                                                          scalar_product(self.a, self.v)))
        def test_distributive_field(self):
            np.testing.assert_array_equal(scalar_product(self.a + self.b, self.v),
                                          vector_addition(scalar_product(self.a, self.v), 
                                                          scalar_product(self.b, self.v)))
    run_tests(TestScalarProduct)

def test_inner_product(inner_product, scalar_product):
    class TestInnerProduct(tst.TestCase):
        a = 4
        u = np.array((1, -2, 3))
        v = np.array((-5, 6, -7))
        w = np.array((4,8))
        z = np.zeros(3)

        def test_valid_product(self):
            np.testing.assert_equal(inner_product(self.u, self.v), -38)
        def test_invalid_product(self):
            with self.assertRaises(DimensionError):
                inner_product(self.v, self.w)
        def test_zero(self):
            np.testing.assert_equal(inner_product(self.v, self.z), 0)
        def test_positive_definite(self):
            np.testing.assert_array_less(0, inner_product(self.v, self.v))
        def test_commutative(self):
            np.testing.assert_equal(inner_product(self.u, self.v), inner_product(self.v, self.u))
        def test_linear1(self):
            np.testing.assert_equal(inner_product(scalar_product(self.a, self.v), self.u), 
                                    self.a * inner_product(self.v, self.u))
        def test_linear2(self):
            np.testing.assert_equal(inner_product(self.v, scalar_product(self.a, self.u)), 
                                    self.a * inner_product(self.v, self.u))
    run_tests(TestInnerProduct)

def test_vector_matrix_product(matrix_product):
    class TestVectorMatrixProduct(tst.TestCase):
        a = 4
        v = np.array((1, 2, 3))
        M = np.array(((6, 5, 4), (9, 8, 7))) 
        z = np.zeros(3)

        def test_valid_product(self):
            np.testing.assert_array_equal(matrix_product(self.M, self.v), np.array((28, 46)))
        def test_invalid_product(self):
            with self.assertRaises(DimensionError):
                matrix_product(self.M.T, self.v)
        def test_scalar_product(self):
            np.testing.assert_array_equal(matrix_product(2*self.M, self.v), 2*matrix_product(self.M, self.v))
        def test_scalar_product2(self):
            np.testing.assert_array_equal(matrix_product(self.M, 2*self.v), 2*matrix_product(self.M, self.v))
    run_tests(TestVectorMatrixProduct)

def test_matrix_product(matrix_product):
    class TestMatrixProduct(tst.TestCase):
        a = 4
        v = np.array((1, 2, 3))
        M = np.array(((6, 5, 4), (9, 8, 7))) 
        N = np.array(((4, 0), (4, 5), (8, 3))) 
        z = np.zeros(3)

        def test_vector_product(self):
            np.testing.assert_array_equal(matrix_product(self.M, self.v).reshape(2), np.array((28, 46)))
        def test_matrix_product(self):
            np.testing.assert_array_equal(matrix_product(self.M, self.N), np.array(((76, 37), (124, 61))))
        def test_invalid_product(self):
            with self.assertRaises(DimensionError):
                matrix_product(self.M, self.M)
    run_tests(TestMatrixProduct)

def test_neural_network(read_network, run_network, matrix_product):
    class TestNeuralNetwork(tst.TestCase):

        layer1 = np.array(((0.5,  0.2,  0  ,  0  , -0.2),
                           (0.2, -0.5, -0.1,  0.9, -0.8),
                           (0  ,  0.2,  0  ,  0.1, -0.1),
                           (0.1,  0.8,  0.3,  0  , -0.7)))
        layer2 = np.array(((0.5,  0.2, -0.1,  0.9),
                           (0.2, -0.5,  0.3,  0.1)))

        def test_read_singlelayer(self):
            np.testing.assert_array_almost_equal(read_network(r"example.json"), self.layer1)
        def test_read_duallayer(self):
            np.testing.assert_array_almost_equal(read_network(r"example-2layer.json"), 
                                                 matrix_product(self.layer2, self.layer1))
        def test_run_singlelayer(self):
            np.testing.assert_array_almost_equal(run_network(r"example.json", np.ones(5)).reshape(4), 
                                                 np.array((0.5, -0.3, 0.2, 0.5)))
        def test_run_duallayer(self):
            np.testing.assert_array_almost_equal(run_network(r"example-2layer.json", np.ones(5)).reshape(2), 
                                                 np.array((0.62, 0.36)))
    run_tests(TestNeuralNetwork)

def test_determinant(determinant_2, determinant_3, determinant):
    class TestDeterminant(tst.TestCase):
        M = np.array(((1,2),(3,4)))
        N = np.array(((1,-2,0),(0,1,-1),(1,-5,1)))
        O = np.array(((1,2),(3,4),(5,6)))
        P = np.array(((1,-1),(2,-2)))
        Q = np.array(((1,2,3),(4,5,6),(7,8,9)))
        R = np.array(((1,2,3,4),(5,6,7,8),(9,10,11,12)))

        def test_determinant2_nonzero(self):
            np.testing.assert_equal(determinant_2(self.M), -2)
        def test_determinant2_zero(self):
            np.testing.assert_equal(determinant_2(self.P), 0)
        def test_determinant2_invalid(self):
            with self.assertRaises(DimensionError):
                determinant_2(self.N)

        def test_determinant3_nonzero(self):
            np.testing.assert_equal(determinant_3(self.N), -2)
        def test_determinant3_zero(self):
            np.testing.assert_equal(determinant_3(self.Q), 0)
        def test_determinant3_invalid(self):
            with self.assertRaises(DimensionError):
                determinant_3(self.M)

        def test_determinant_nonzero2(self):
            np.testing.assert_equal(determinant(self.M), -2)
        def test_determinant_nonzero3(self):
            np.testing.assert_equal(determinant(self.N), -2)
        def test_determinant_zero2(self):
            np.testing.assert_equal(determinant(self.P), 0)
        def test_determinant_zero3(self):
            np.testing.assert_equal(determinant(self.Q), 0)
        def test_determinant_scalar(self):
            np.testing.assert_equal(determinant(np.array(((42)))), 42)
        def test_determinant_too_large(self):
            with self.assertRaises(DimensionError):
                determinant(self.R)
        def test_determinant_invalid(self):
            with self.assertRaises(DimensionError):
                determinant(self.O)
    run_tests(TestDeterminant)

def test_inverse_2(inverse_matrix_2):
    class TestInverse2(tst.TestCase):
        M = np.array(((1,2),(3,4)))
        N = np.array(((1,-1),(2,-2)))
        O = np.array(((1,2),(3,4),(5,6)))

        def test_inverse2(self):
            np.testing.assert_equal(inverse_matrix_2(self.M), np.array(((-2,1),(1.5,-0.5))))
        def test_inverse2_no_inverse(self):
            with self.assertRaises(NonInvertibleError):
                inverse_matrix_2(self.N)
        def test_inverse2_invalid(self):
            with self.assertRaises(DimensionError):
                inverse_matrix_2(self.O)
    run_tests(TestInverse2)

def test_inverse_(inverse_matrix):
    class TestInverse(tst.TestCase):
        M = np.array(((4,6,3,4),(4,7,2,6),(3,3,3,1),(3,7,1,6)))
        Mi = np.array(((-11,3,8,3),(-2,-1,2,2),(11,-2,-8,-4),(6,0,-5,-3)))
        N = np.array(((1,2,3),(4,5,6),(7,8,9)))
        O = np.array(((1,2),(3,4),(5,6)))

        def test_inverse(self):
            np.testing.assert_equal(inverse_matrix(self.M), self.Mi)
        def test_inverse_no_inverse(self):
            with self.assertRaises(NonInvertibleError):
                inverse_matrix(self.N)
        def test_inverse_invalid(self):
            with self.assertRaises(DimensionError):
                inverse_matrix(self.O)
    run_tests(TestInverse)

def test_magisch_vierkant(magisch_vierkant):
    class TestMagischVierkant(tst.TestCase):
        i = np.array(((0, 3, 4), (0, 0, 0), (0, 7, 0)))
        i2 = np.array(((8, 3, 4), (1, 5, 9), (6, 7, 2)))
        r = np.array(((5, 0, 0), (0, 0, 4), (0, 0, 6)))
        r2 = np.array(((5, 5, 6.5), (7, 5.5, 4), (4.5, 6, 6)))

        def test_integer(self):
            np.testing.assert_equal(magisch_vierkant(self.i), self.i2)
        def test_rational(self):
            np.testing.assert_equal(magisch_vierkant(self.r), self.r2)
    run_tests(TestMagischVierkant)

def test_limit(limit_left, limit_right, limit):
    class TestLimit(tst.TestCase):
        def discontinuous_function(self, x: float) -> float:
            if x == 72:
                return -10
            elif x % 13 == 0:
                return None
            else:
                return 2.5 * x

        def holes_function(self, x: float) -> float:
            if x % 13 == 0:
                return None
            else:
                return 2.5 * x

        def single_discontinuity_function(self, x: float) -> float:
            if x == 72:
                return -10
            else:
                return 2.5 * x

        def right_undefined_function(self, x: float) -> float:
            if x >= 10:
                return None
            else:
                return x+3

        def left_undefined_function(self, x: float) -> float:
            if x <= 10:
                return None
            else:
                return x+3

        def piecewise_function(self, x: float) -> float:
            if x < -2:
                return -1.5*x -2
            elif x >= -2 and x <= 1:
                return -1/3 * (x-1) + 2
            else:
                return x-2

        def test_single_discontinuity(self):
            np.testing.assert_equal(limit(self.single_discontinuity_function,72), 180)
        def test_holes(self):
            np.testing.assert_equal(limit(self.holes_function,78), 195)
        def test_left_undefined_below(self):
            np.testing.assert_almost_equal(limit_left(self.right_undefined_function,10), 13, 3)
        def test_left_undefined_above(self):
            np.testing.assert_almost_equal(limit_right(self.left_undefined_function,10), 13, 3)
        def test_jump_left(self):
            np.testing.assert_almost_equal(limit_left(self.piecewise_function,1), 2, 3)
        def test_jump_right(self):
            np.testing.assert_almost_equal(limit_right(self.piecewise_function,1), -1, 3)
        def test_jump(self):
            np.testing.assert_equal(limit(self.piecewise_function,1), None)

    run_tests(TestLimit)

def test_numeric_derivative(get_derivative_at):
    class TestNumericDerivative(tst.TestCase):
        def square(self, x: float) -> float:
            return x**2

        def double(self, x: float) -> float:
            return x*2

        def succ(self, x: float) -> float:
            return x+1

        def test_square(self):
            np.testing.assert_almost_equal(get_derivative_at(self.square, 2), 4, 3)
        def test_double(self):
            np.testing.assert_almost_equal(get_derivative_at(self.double, 2), 2, 3)
        def test_succ(self):
            np.testing.assert_almost_equal(get_derivative_at(self.succ, 2), 1, 3)

    run_tests(TestNumericDerivative)

def test_verkeer_snelheden(get_data, bereken_deltas):
    class TestVerkeerSnelheden(tst.TestCase):
        time, car1, car2 = get_data()

        def test_min_1(self):
            car1_speed = bereken_deltas(self.time, self.car1)
            np.testing.assert_almost_equal(min(car1_speed), 1.27)
        def test_max_1(self):
            car1_speed = bereken_deltas(self.time, self.car1)
            np.testing.assert_almost_equal(max(car1_speed), 6.35)
        def test_min_2(self):
            car2_speed = bereken_deltas(self.time, self.car2)
            np.testing.assert_almost_equal(min(car2_speed), 0)
        def test_max_2(self):
            car2_speed = bereken_deltas(self.time, self.car2)
            np.testing.assert_almost_equal(max(car2_speed), 2)
    run_tests(TestVerkeerSnelheden)

def test_polynomial_derivative(get_derivative):
    class TestNumericDerivative(tst.TestCase):
        from ac import polynomial
        x_squared = polynomial({1: 0, 2: 1})
        x_recip = polynomial({-1: 1})
        x_root = polynomial({1/2: 1})

        def test_squared(self):
            np.testing.assert_equal(get_derivative(self.x_squared)[0][1], 2)
        def test_primes(self):
            np.testing.assert_equal(get_derivative(self.x_squared)[3], 1)
        def test_recip(self):
            np.testing.assert_equal(get_derivative(self.x_recip)[0][-2], -1)
        def test_root(self):
            np.testing.assert_equal(get_derivative(self.x_root)[0][-0.5], 0.5)

    run_tests(TestNumericDerivative)

def test_matrix_derivative(deriv_matrix, matrix_product):
    fx = np.array((2,1,3))
    class TestMatrixDerivative(tst.TestCase):

        def test_derivative(self):
            np.testing.assert_array_equal(deriv_matrix(fx).flatten(), np.array((1,6,0)))
    run_tests(TestMatrixDerivative)

def test_symbolic_differentiation_alfa(Constant, Variable, Sum, Product, Power):
    class TestSymbolicDifferentiationAlfa(tst.TestCase):

        def test_variable(self):
            form = Function('f', Variable('x'))
            deriv = Function('f', Constant(1), 1)
            np.testing.assert_equal(form.deriv(), deriv)
        def test_product(self):
            form = Function('f', Product(Variable('x'), Variable('x')))
            deriv = Function('f', Sum(Variable('x'),Variable('x')) ,1)
            np.testing.assert_equal(form.deriv(), deriv)
        def test_2x_plus_3(self):
            form = Function('f', Sum(Product(Constant(2), Power(Variable('x'),1)), Constant(3)))
            deriv = Function('f', Constant(2), 1)
            np.testing.assert_equal(form.deriv(), deriv)
        def test_recip_x(self):
            form = Function('f', Power(Variable('x'), -1))
            deriv = Function('f',Negative(Power(Variable('x'),-2)),1)
            np.testing.assert_equal(form.deriv(), deriv)
    run_tests(TestSymbolicDifferentiationAlfa)

def test_symbolic_differentiation_bravo(Constant, Variable, Sum, Product, Power, Sin, Cos, Tan):
    class TestSymbolicDifferentiationBravo(tst.TestCase):

        def test_sin(self):
            form = Function('f', Sin(Variable('x')))
            deriv = Function('f', Cos(Variable('x')), 1)
            np.testing.assert_equal(form.deriv(), deriv)
        def test_cos(self):
            form = Function('f', Cos(Variable('x')))
            deriv = Function('f',Negative(Sin(Variable('x'))),1)
            np.testing.assert_equal(form.deriv(), deriv)
        def test_tan(self):
            form = Function('f', Tan(Variable('x')))
            deriv = Function('f',Power(Sec(Variable('x')),2),1)
            np.testing.assert_equal(form.deriv(), deriv)
    run_tests(TestSymbolicDifferentiationBravo)

def test_symbolic_differentiation_charlie(Constant, Variable, Sum, Product, Power, Sin, Cos, Tan, E, Exponent, Ln, Log):
    class TestSymbolicDifferentiationCharlie(tst.TestCase):

        def test_e(self):
            form = Function('f', E(Variable('x')))
            deriv = Function(label='f',body=E(exponent=Variable(label='x')),deriv_order=1)
            np.testing.assert_equal(form.deriv(), deriv)
        def test_exponent(self):
            form = Function('f', Exponent(Constant(2), Variable('x')))
            deriv = Function(label='f',body=Product(left=Exponent(base=Constant(value=2),exponent=Variable(label='x')),right=Ln(argument=Constant(value=2))),deriv_order=1)
            np.testing.assert_equal(form.deriv(), deriv)
        def test_ln(self):
            form = Function('f', Ln(Variable('x')))
            deriv = Function(label='f',body=Power(base=Variable(label='x'),exponent=-1),deriv_order=1)
            np.testing.assert_equal(form.deriv(), deriv)
        def test_log(self):
            form = Function('f', Log(Constant(2), Variable('x')))
            deriv = Function(label='f',body=Power(base=Product(left=Variable(label='x'),right=Ln(argument=Constant(value=2))),exponent=-1),deriv_order=1)
            np.testing.assert_equal(form.deriv(), deriv)
    run_tests(TestSymbolicDifferentiationCharlie)

def test_symbolic_differentiation_charlie_eq(Constant, Variable, Sum, Product, Power, Sin, Cos, Tan, E, Exponent, Ln, Log):
    class TestSymbolicDifferentiationCharlieEq(tst.TestCase):
        def test_e_equivalent(self):
            form = Function('f', E(Variable('x')))
            np.testing.assert_almost_equal(form.deriv().eval({'x':-1}), 0.368, 3)
        def test_exponent_equivalent(self):
            form = Function('f', Exponent(Constant(2), Variable('x')))
            np.testing.assert_almost_equal(form.deriv().eval({'x':7}), 88.723, 3)
        def test_ln_equivalent(self):
            form = Function('f', Ln(Variable('x')))
            np.testing.assert_almost_equal(form.deriv().eval({'x':3}), 0.333, 3)
        def test_log_equivalent(self):
            form = Function('f', Log(Constant(2), Variable('x')))
            np.testing.assert_almost_equal(form.deriv().eval({'x':5}), 0.289, 3)
    run_tests(TestSymbolicDifferentiationCharlieEq)

def test_symbolic_differentiation_delta(Constant, Variable, Sum, Product, Power, Sin, Cos, Tan, E, Exponent, Ln, Log):
    class TestSymbolicDifferentiationDelta(tst.TestCase):

        def test_e_x_squared(self):
            form = Function('f', E(Power(Variable('x'),2)))
            deriv = Function(label='f',body=Product(left=Product(left=Constant(value=2),right=Variable(label='x')),right=E(exponent=Power(base=Variable(label='x'),exponent=2))),deriv_order=1)
            np.testing.assert_equal(form.deriv(), deriv)
        def test_five_log_e_x(self):
            form = Function('f', Exponent(Constant(5), E(Variable('x'))))
            deriv = Function(label='f',body=Product(left=E(exponent=Variable(label='x')),right=Product(left=Exponent(base=Constant(value=5),exponent=E(exponent=Variable(label='x'))),right=Ln(argument=Constant(value=5)))),deriv_order=1)
            np.testing.assert_equal(form.deriv(), deriv)
        def test_ln_x_squared(self):
            form = Function('f', Ln(Power(Variable('x'),2)))
            deriv = Function(label='f',body=Product(left=Constant(value=2),right=Power(base=Variable(label='x'),exponent=-1)),deriv_order=1)
            np.testing.assert_equal(form.deriv(), deriv)
        def test_five_to_the_e_x(self):
            form = Function('f', Log(Constant(5), E(Variable('x'))))
            deriv = Function(label='f',body=Power(base=Ln(argument=Constant(value=5)),exponent=-1),deriv_order=1)
            np.testing.assert_equal(form.deriv(), deriv)
        def test_sin_squared_x(self):
            form = Function('f', Power(Sin(Variable('x')), 2))
            deriv = Function(label='f',body=Product(left=Product(left=Constant(value=2),right=Cos(argument=Variable(label='x'))),right=Sin(argument=Variable(label='x'))),deriv_order=1)
            np.testing.assert_equal(form.deriv(), deriv)
    run_tests(TestSymbolicDifferentiationDelta)

def test_symbolic_differentiation_delta_eq(Constant, Variable, Sum, Product, Power, Sin, Cos, Tan, E, Exponent, Ln, Log):
    class TestSymbolicDifferentiationDeltaEq(tst.TestCase):

        def test_e_x_squared_equivalent(self):
            form = Function('f', E(Power(Variable('x'),2)))
            np.testing.assert_almost_equal(form.deriv().eval({'x':5}), 720048993373.859, 3)
        def test_five_log_e_x_equivalent(self):
            form = Function('f', Exponent(Constant(5), E(Variable('x'))))
            np.testing.assert_almost_equal(form.deriv().eval({'x':-1}), 1.07, 3)
        def test_ln_x_squared_equivalent(self):
            form = Function('f', Ln(Power(Variable('x'),2)))
            np.testing.assert_equal(form.deriv().eval({'x':8}), 0.25)
        def test_five_to_the_e_x_equivalent(self):
            form = Function('f', Log(Constant(5), E(Variable('x'))))
            np.testing.assert_almost_equal(form.deriv().eval({'x':3}), 0.621, 3)
        def test_sin_squared_x_equivalent(self):
            form = Function('f', Power(Sin(Variable('x')), 2))
            np.testing.assert_almost_equal(form.deriv().eval({'x': 1.5 * math.pi}), 0, 3)
    run_tests(TestSymbolicDifferentiationDeltaEq)

def test_regressie(train, gradient, cost, data):
    class TestRegressie(tst.TestCase):

        def test_convergence(self):
            s, i = train(data)
            np.testing.assert_array_less(cost(data, s, i), 3200)
    run_tests(TestRegressie)

def test_verkeer_posities(get_data, bereken_posities, vind_botsing):
    class TestVerkeerPosities(tst.TestCase):
        time, car1, car2, car3 = get_data()
        car1_position = bereken_posities(time, car1)
        car2_position = bereken_posities(time, car2)
        car3_position = bereken_posities(time, car3)

        (t,ca,cap,cb,cbp) = vind_botsing(time,car1_position,car2_position,car3_position)

        def test_time(self):
            np.testing.assert_almost_equal(self.t, 0.4, 0.1)
        def test_car_a(self):
            np.testing.assert_equal(self.ca, 1)
        def test_car_a_pos(self):
            np.testing.assert_equal(self.cb, 3)
        def test_car_b(self):
            np.testing.assert_almost_equal(self.cap, 2.7, 0.01)
        def test_car_b_pos(self):
            np.testing.assert_almost_equal(self.cbp, 1.9, 0.01)

    run_tests(TestVerkeerPosities)

def test_symbolic_integration_alfa(Constant, Variable, Sum, Product, Power):
    class TestSymbolicIntegrationAlfa(tst.TestCase):

        def test_variable_x(self):
            form = Function('f', Variable('x'))
            integral = Function('f', Sum(Product(Constant(0.5),Power(Variable('x'),2)),Variable('C')), -1)
            np.testing.assert_equal(form.integrate('x'), integral)
        def test_variable_y(self):
            form = Function('f', Variable('y'))
            integral = Function('f', Sum(Product(Variable('x'),Variable('y')),Variable('C')), -1)
            np.testing.assert_equal(form.integrate('x'), integral)
        def test_product(self):
            form = Function('f', Product(Variable('x'), Variable('y')))
            integral = Function('f', Sum(Product(Variable('y'),Product(Constant(0.5),Power(Variable('x'),2))),Variable('C')), -1)
            np.testing.assert_equal(form.integrate('x'), integral)
        def test_sum(self):
            form = Function('f', Sum(Variable('x'), Variable('y')))
            integral = Function('f', Sum(Sum(Product(Constant(0.5),Power(Variable('x'),2)),Product(Variable('x'),Variable('y'))),Variable('C')), -1)
            np.testing.assert_equal(form.integrate('x'), integral)
        def test_power(self):
            form = Function('f', Power(Variable('x'), 3))
            integral = Function('f',Sum(Product(Constant(0.25),Power(Variable('x'),4)),Variable('C')),-1)
            np.testing.assert_equal(form.integrate('x'), integral)

    run_tests(TestSymbolicIntegrationAlfa)

def test_symbolic_integration_alfa_eq(Constant, Variable, Sum, Product, Power):
    class TestSymbolicIntegrationAlfaEq(tst.TestCase):

        def test_variable_x_equivalent(self):
            form = Function('f', Variable('x'))
            np.testing.assert_equal(form.integrate('x').eval({'x': 12}), 72)
        def test_variable_y_equivalent(self):
            form = Function('f', Variable('y'))
            np.testing.assert_equal(form.integrate('x').eval({'x': 3, 'y': 4}), 12)
        def test_product_equivalent(self):
            form = Function('f', Product(Variable('x'), Variable('y')))
            np.testing.assert_equal(form.integrate('x').eval({'x': 8, 'y': 2}), 64)
        def test_sum_equivalent(self):
            form = Function('f', Sum(Variable('x'), Variable('y')))
            np.testing.assert_equal(form.integrate('x').eval({'x': 1, 'y': 9}), 9.5)
        def test_power_equivalent(self):
            form = Function('f', Power(Variable('x'), 3))
            np.testing.assert_equal(form.integrate('x').eval({'x': 11}), 3660.25)

    run_tests(TestSymbolicIntegrationAlfaEq)

def test_symbolic_integration_bravo(Constant, Variable, Sum, Product, Power, Sin, Cos, Tan, E, Exponent, Ln, Log):
    class TestSymbolicIntegrationBravo(tst.TestCase):

        def test_sin(self):
            form = Function('f', Sin(Variable('x')))
            integral = Function('f',Sum(Negative(Cos(Variable('x'))),Variable('C')),-1)
            np.testing.assert_equal(form.integrate('x'), integral)
        def test_cos(self):
            form = Function('f', Cos(Variable('x')))
            integral = Function('f',Sum(Sin(Variable('x')),Variable('C')),-1)
            np.testing.assert_equal(form.integrate('x'), integral)
        def test_tan(self):
            form = Function('f', Tan(Variable('x')))
            integral = Function('f',Sum(Negative(Ln(Cos(Variable('x')))),Variable('C')),-1)
            np.testing.assert_equal(form.integrate('x'), integral)
        def test_e(self):
            form = Function('f', E(Variable('x')))
            integral = Function('f',Sum(E(Variable('x')),Variable('C')),-1)
            np.testing.assert_equal(form.integrate('x'), integral)
        def test_exponent(self):
            form = Function('f', Exponent(Constant(3), Variable('x')))
            integral = Function('f',Sum(Product(Exponent(Constant(3),Variable('x')),
                                                Power(Ln(Constant(3)),-1)),Variable('C')),-1)
            np.testing.assert_equal(form.integrate('x'), integral)
        def test_ln(self):
            form = Function('f', Ln(Variable('x')))
            integral = Function('f',Sum(Product(Variable('x'),Sum(Ln(Variable('x')),
                         Negative(Constant(1)))),Variable('C')),-1)
            np.testing.assert_equal(form.integrate('x'), integral)
        def test_log(self):
            form = Function('f', Log(Constant(3),Variable('x')))
            integral = Function('f',Sum(Product(Product(Variable('x'),Sum(Ln(Variable('x')),
                         Negative(Constant(1)))),Power(Ln(Constant(3)),-1)),Variable('C')),-1)
            np.testing.assert_equal(form.integrate('x'), integral)

    run_tests(TestSymbolicIntegrationBravo)


def test_symbolic_integration_bravo_eq(Constant, Variable, Sum, Product, Power, Sin, Cos, Tan, E, Exponent, Ln, Log):
    class TestSymbolicIntegrationBravoEq(tst.TestCase):
        def test_sin_equivalent(self):
            form = Function('f', Sin(Variable('x')))
            np.testing.assert_almost_equal(form.integrate('x').eval({'x': -1}), -0.54,3)
        def test_cos_equivalent(self):
            form = Function('f', Cos(Variable('x')))
            np.testing.assert_almost_equal(form.integrate('x').eval({'x': 7}), 0.657,3)
        def test_tan_equivalent(self):
            form = Function('f', Tan(Variable('x')))
            np.testing.assert_almost_equal(form.integrate('x').eval({'x': 13}), 0.097,3)
        def test_e_equivalent(self):
            form = Function('f', E(Variable('x')))
            np.testing.assert_almost_equal(form.integrate('x').eval({'x': 5}), 148.413,3)
        def test_exponent_equivalent(self):
            form = Function('f', Exponent(Constant(3), Variable('x')))
            np.testing.assert_almost_equal(form.integrate('x').eval({'x': 7}), 1990.693,3)
        def test_ln_equivalent(self):
            form = Function('f', Ln(Variable('x')))
            np.testing.assert_almost_equal(form.integrate('x').eval({'x': 42}), 114.982,3)
        def test_log_equivalent(self):
            form = Function('f', Log(Constant(3),Variable('x')))
            np.testing.assert_almost_equal(form.integrate('x').eval({'x': 72}), 214.744,3)

    run_tests(TestSymbolicIntegrationBravoEq)

def test_euler(afgeleide_functie, euler):
    class TestEuler(tst.TestCase):

        def test_result(self):
            np.testing.assert_almost_equal(euler(afgeleide_functie, 0, 1, 0.025, 0.95), 129.876, 3)

    run_tests(TestEuler)
