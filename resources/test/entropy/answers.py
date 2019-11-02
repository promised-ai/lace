import numpy as np
import scipy as sp
import itertools as it

from scipy.special import logsumexp
from scipy.stats import norm
from scipy.integrate import quad


class Categorical(object):
    def __init__(self, ln_weights):
        self.ln_weights = ln_weights

    def logpdf(self, x):
        return self.ln_weights[x]

    def pdf(self, x):
        return np.exp(self.logpdf(x))


class Mixture(object):
    def __init__(self, weights, components):
        self.weights = weights
        self.ln_weights = np.log(weights)
        self.components = components

    def logpdf(self, x):
        parts = [w + c.logpdf(x) for w, c in zip(self.ln_weights, self.components)]
        return logsumexp(parts)

    def pdf(self, x):
        return sum(w * c.pdf(x) for w, c in zip(self.weights, self.components))


class Product(object):
    def __init__(self, components):
        self.components = components

    def logpdf(self, xs):
        return sum(c.logpdf(x) for x, c in zip(xs, self.components))

    def pdf(self, xs):
        return np.exp(self.logpdf(xs))


if __name__ == "__main__":
    # Categorical - Categorical
    # -------------------------
    # categorical-categorical - state 0
    c21 = Categorical([-2.56494936, -1.87180218, -0.77318989, -1.178655])
    c22 = Categorical([-0.91629073, -1.2039728, -1.60943791, -2.30258509])
    cm_s1v1 = Mixture([0.5, 0.5], [c21, c22])

    c31 = Categorical([-0.69314718, -0.69314718])
    c32 = Categorical([-0.91629073, -0.51082562])
    cm_s1v2 = Mixture([0.25, 0.75], [c31, c32])


    cat_1 = Product([cm_s1v1, cm_s1v2])
    hc0 = 0.0
    ps_hc0 = []
    for x, y in it.product(range(4), range(2)):
        p = cat_1.pdf([x, y])
        ps_hc0.append(p)
        # print("px({}) = {}, py({}) = {}, p = {}".format(x, np.log(px), y, np.log(py), np.log(p)))
        hc0 -= p * np.log(p)

    print("state 0 H(X,Y): %f" % hc0)

    # categorical-categorical - state 1
    cp1 = Product([c21, c31])
    cp2 = Product([c22, c32])
    cat_2 = Mixture([0.25, 0.75], [cp1, cp2])

    hc1 = 0.0
    ps_hc1 = []
    for x, y in it.product(range(4), range(2)):
        p = cat_2.pdf([x, y])
        ps_hc1.append(p)
        # print("pxy({}, {}) = {}".format(x, y, np.log(p)))
        hc1 -= p * np.log(p)

    print("state 1 H(X,Y): %f" % hc1)

    # categorical-categorical - both states
    cat_all = Mixture([0.5, 0.5], [cat_1, cat_2])
    h = 0.0
    for p1, p2 in zip(ps_hc0, ps_hc1):
        p = (p1 + p2)/2.0
        h -= p*np.log(p)

    print("H(X,Y): %f" % h)


    # Categorical - Gaussian
    # ----------------------
    # Cat column 2 and Gaussian column 0 - state 0
    g01 = norm(0.0, 1.0);
    g02 = norm(-0.8, 0.75);
    gm_s1v1 = Mixture([0.5, 0.5], [g01, g02])

    gp1 = Product([c21, g01])
    gp2 = Product([c22, g02])
    gpm = Mixture([0.5, 0.5], [gp1, gp2])

    h_gc = 0.0
    for y in range(4):
        def fn(x):
            p = gpm.pdf([y, x])
            return -p*np.log(p)
        q = quad(fn, -20.0, 20.0)
        # print(q)
        h_gc += q[0]

    print(h_gc)

    # Cat column 2 and Gaussian column 0 - state 1
    cm_s2v1 = Mixture([0.25, 0.75], [c21, c22])
    gauss_2 = Product([cm_s2v1, gm_s1v1])
    h_gc = 0.0
    for y in range(4):
        def fn(x):
            p = gauss_2.pdf([y, x])
            return -p*np.log(p)

        q = quad(fn, -20.0, 20.0)

        h_gc += q[0]

    print(h_gc)

    cg_all = Mixture([0.5, 0.5], [gpm, gauss_2])
    h_gc = 0.0
    for y in range(4):
        def fn(x):
            p = cg_all.pdf([y, x])
            return -p*np.log(p)

        q = quad(fn, -20.0, 20.0)

        h_gc += q[0]

    print(h_gc)


