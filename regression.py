# from numpy import linalg
# from numpy import mat
# from numpy import shape
# from numpy import zeros
# from numpy import arange

import numpy as np

# Create matrix X in Xw = Y
def createX(x, m):
    X = np.zeros(shape=((len(x), m + 1))) # create matrix X from orde-m (m+1)
    row_index = 0
    for x_elem in x:
        xrow = []
        for i in range(m + 1):
            xrow.append(pow(x_elem, i))
        X[row_index:] = xrow
        row_index += 1
        # print 'rank: ', np.rank(np.mat(X))
    return np.mat(X)  # return matrix instead of ndarray


def standReg(x, y, m):
    X = createX(x, m)  # create X
    y = np.mat(y) # treat y as matrix object
    y = y.T  # prepare y -> transpose the matrix
    XTX = X.T * X # Xtranspose * X
    if np.linalg.det(XTX) == 0.0: # find determinant Xtranspose * X
        print "The XTX matrix is singular, cannot do inverse"
        return
    w = XTX.I * X.T * y # normal equation / LR Normal
    print 'w_stand: ', w # print w found
    return w


def dummyReg(x, y, m):
    X = createX(x, m)  # create X
    y = np.mat(y)
    y = y.T  # prepare y
    # if np.linalg.det(X) == 0.0:
    #     print "The X matrix is singular, cannot do inverse"
    #     return
    if np.rank(X) == 0.0:
        print "The X matrix has zero rank"
        return
    w = X.I * y
    print 'w_dummy: ', w
    return w


def numpyReg(x, y, m):
    X = createX(x, m)  # create X
    y = np.mat(y)
    y = y.T # prepare y
    ls = np.linalg.lstsq(X, y)
    # print 'ls: ', ls
    print 'w_numpy', ls[0]
    return ls[0]


def ridgeReg(x, y, m, lam=0.2):
    X = createX(x, m)  # create X
    y = np.mat(y)
    y = y.T  # prepare y
    XTX = X.T * X
    n = np.shape(XTX)[0]
    XTX = XTX + np.eye(n) * lam
    if np.linalg.det(XTX) == 0.0:
        print "The XTX matrix is singular, cannot do inverse"
        return
    w = XTX.I * X.T * y
    print 'w_ridge: ', w
    return w


def tikhonovReg(x, y, m):
    X = createX(x, m)  # create X
    y = np.mat(y)
    y = y.T  # prepare y
    XTX = X.T * X
    n = np.shape(XTX)[0]
    XTX = XTX + np.eye(n).T * np.eye(n)
    if np.linalg.det(XTX) == 0.0:
        print "The XTX matrix is singular, cannot do inverse"
        return
    w = XTX.I * X.T * y
    print 'w_ridge: ', w
    return w

# SVD Regression Own Version
def svdReg(x, y, m):
    """
    SVD = UDV*
    dimana U dan V merupakan unitary matriks yang saling orthogonal
    * = konjugasi matriks. Serupa dengan transpose, tapi elemen imajiner di-invers
    D = matriks diagonal non negatif

    Algorithm taken from http://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm
    """
    X = createX(x, m) # create matrix X from x with order m
    y = np.mat(y) # treat y as matrix
    y = y.T # transpose y

    # the eigenvectors of XXT is the U matrix
    XXT = X * X.T
    # find eigenvalues of XXT
    eigvalXXT, eigvecXXT = np.linalg.eig(XXT) # return eigenvalue; eigenvector
    # print 'EGVEC_U = ', eigvalXXT

    # the eigenvectors of XTX is the V matrix
    XTX = X.T * X
    # find eigenvalues of XTX
    eigvalXTX, eigvecXTX = np.linalg.eig(XTX)
    # print 'EGVEC_V= ', eigvalXTX

    # entry for diagonal D is the square-root of XTX or XXT
    # prefer using XXT coeficient
    D = np.mat(np.eye(len(x),m+1))
    idx = 0 # index as pointer
    for x in eigvalXXT:
        D[idx,idx] = np.sqrt(x) # replace diagonal entry with square root of respective eignevalue

    # check wether decomposition is valid
    temp = eigvecXXT * D * eigvecXTX.T
    print 'TEMP = ', temp
    print 'ORIG = ', X

    # solve the SVD-LR
    # w = V * Dinverse * Utranspose * y
    w = eigvecXTX * D.I * eigvecXXT.T * y

    # return model
    return w

def numpySVDReg(x, y, m):
    X = createX(x, m) # create matrix X from x with order m
    y = np.mat(y) # treat y as matrix
    y = y.T # transpose y

    # compute SVD using numpy linalg library
    U,s,V = np.linalg.svd(X, full_matrices=True)
    S = np.zeros(shape=(len(x), m+1)) # create dummy matrix for S
    offsetmin = min(len(x), m+1) # find minimum value between data length and model order
    S[:offsetmin, :offsetmin] = np.diag(s) # fill diagonal with singular value s
    # solve equation
    # print X
    # print np.dot(U, np.dot(S, V))
    # print U * S * V # Equal to U * D * Vtranspose
    # print np.allclose(X, np.dot(U, np.dot(S, V)))
    w = V.T * np.mat(S).I * U.T * y
    print w
    return w

def createModel(x, w):
    y = np.zeros(len(x)) # create vector y with same length with x
    y = np.mat(y) # treat y as matrix object
    # y = y.T  # prepare y, transpose matrix
    pwr = np.arange(len(w)) # prepare polynom order
    for wi, p in zip(w, pwr):
        accum = wi * np.power(x, p)
        # print 'accum wi = ', wi, ' p = ', p, ' is ', accum
        y = y + accum # accum.T
    return np.squeeze(np.asarray(y.T))
    # return np.asarray(y.T)