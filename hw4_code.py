import numpy as np 
import matplotlib as mlp
import matplotlib.pyplot as plt

#############################################
################ Problem 2 ##################
#############################################

def generate_input():
    a = np.random.randn(500, 100)
    x = np.random.randn(100, 1)
    y = np.random.randn(500, 1)
    y = np.multiply(y, y)
    b = a.dot(x) + y
    c = np.random.randn(100, 1)
    return a, b, c, x

def f(a, b, c, x):
    rv1 = np.sum(c * x)
    bax = b - a.dot(x)
    logbax = np.log(bax)
    return rv1 - np.sum(logbax)

def gradient(a, b, c, x):
    bax = b - a.dot(x)
    term = np.sum(a / bax, axis=0).reshape(100, 1)
    return c + term

def hessian(a, b, c, x):
    bax2 = (b - a.dot(x))**2
    return np.array([[np.sum(a[:,i] * a[:,j] / bax2) for j in range(100)] for i in range(100)])


def backtracking_linesearch(p, x, c, rho, af, bf, cf):
    rv = 1
    fx = f(af, bf, cf, x)
    while (out_of_domain(p, x, rv, af, bf) or f(af, bf, cf, x + rv * p) > (c * rv * (gradient(af, bf, cf, x).T).dot(p) + fx)):
        #print("BTLS, rv = ", rv, rv > 0.00000000001)
        rv *= rho
    #print("returning")
    return rv

def out_of_domain(p, x, rho, af, bf):
    y = x + rho * p 
    diff = bf - af.dot(y)
    return np.sum(diff > 0) < 500

def steepest_descent(x0, e, af, bf, cf, cb, rho):
    xk = x0
    g = gradient(af, bf, cf, xk)
    e2 = e * e
    values = [f(af, bf, cf, x0)]
    while np.sum(g**2) > e2:
        #print("Steepest descent, error = ", np.sum(g**2), len(values))
        p = g * -1
        alpha = backtracking_linesearch(p, xk, cb, rho, af, bf, cf)
        xk += alpha * p 
        values.append(f(af, bf, cf, xk))
        g = gradient(af, bf, cf, xk)
    print(len(values), np.sum(g**2) - e2)
    return values

def newton(x0, e, af, bf, cf, cb, rho):
    xk = x0
    values = [f(af, bf, cf, x0)]
    g = gradient(af, bf, cf, xk)
    hinv = np.linalg.inv(hessian(af, bf, cf, xk))
    lbd2 = ((g.T).dot(hinv)).dot(g)
    double_e = e * 2
    while lbd2 > double_e:
        print("Newton, error = ", lbd2 - double_e)
        p = -hinv.dot(g)
        alpha = backtracking_linesearch(p, xk, cb, rho, af, bf, cf)
        xk += alpha * p
        values.append(f(af, bf, cf, xk))
        g = gradient(af, bf, cf, xk)
        hinv = np.linalg.inv(hessian(af, bf, cf, xk))
        lbd2 = ((g.T).dot(hinv)).dot(g)
    return values

#############################################
################ Problem 4 ##################
#############################################
                       
def newton_inverse(a):
    m, n = a.shape
    if m != n:
        print("Error: A is not symmetric")
        exit(-1)
    alpha = 1 / np.sum(a * a)
    x = alpha * (a.T)
    i = np.identity(m)
    i2 = i * 2
    while np.linalg.norm(i - a.dot(x)) > 0.00000000001:
        x = x.dot((i2 - a.dot(x)))
    return x
