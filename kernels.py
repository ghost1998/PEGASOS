import numpy as np

def polynomialkernel(x1, x2, d =3):
    return np.power(x1.dot(x2) + 1, d)


def laplacerbfkernel(x1, x2, sigma = 0.1):
    return np.exp((-1/sigma) * np.linalg.norm(x1-x2))


def sigmoidkernel(x1, x2, alpha=0.1, c=0):
    return np.tanh(alpha * x1.dot(x2) + c)

