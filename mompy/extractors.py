#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
"""
Given a moment matrix and its values, extract solutions
"""

import numpy as np
import scipy as sc
import scipy.linalg # for schur decomp, which np doesnt have
import numpy.linalg # for its norm, which suits us better than scipy
import itertools
import ipdb

"""
Various utility methods
"""
import numpy as np
import scipy as sc
import operator
from itertools import chain
from sympy import grevlex
from numpy import array, zeros, diag, sqrt
from numpy.linalg import eig, inv, svd
import scipy.sparse, scipy.stats
import ipdb
from munkres import Munkres
import sys
from functools import reduce

eps = 1e-8

def norm(arr):
    """
    Compute the sparse norm
    """
    if isinstance(arr, sc.sparse.base.spmatrix):
        return sqrt((arr.data**2).sum())
    else:
        return sqrt((arr**2).sum())


def tuple_add(t1, t2):
    """Elementwise addition of two tuples"""
    return tuple( t1[i] + t2[i] for i in range(len(t1)) )

def tuple_diff(t1, t2):
    """Elementwise addition of two tuples"""
    return tuple( t1[i] - t2[i] for i in range(len(t1)) )

def tuple_min(t1, t2):
    """Return the entry-wise minimum of the two tuples"""
    return tuple(min(a, b) for a, b in zip(t1, t2))

def tuple_max(t1, t2):
    """Return the entry-wise maximum of the two tuples"""
    return tuple(max(a, b) for a, b in zip(t1, t2))

def tuple_incr(t1, idx, val=1):
    """Return a tuple with the index idx incremented by val"""
    return t1[:idx] + (t1[idx]+val,) + t1[idx+1:]

def tuple_border(t):
    """Return the border of a tuple"""
    return [tuple_incr(t, i) for i in range(len(t))]

def tuple_subs(t1, t2):
    """
    Does t1_i > t2_i for all i?
    """
    d = tuple_diff(t1, t2)
    if any(i < 0 for i in d):
        return False
    else:
        return True

def nonzeros(lst):
    """Return non-zero indices of a list"""
    return (i for i in range(len(lst)) if lst[i] > 0)

def first(iterable, default=None, key=None):
    """
    Return the first element in the iterable
    """
    if key is None:
        for el in iterable:
            return el
    else:
        for key_, el in iterable:
            if key == key_:
                return el
    return default

def prod(iterable):
    """Get the product of elements in the iterable"""
    return reduce(operator.mul, iterable, 1)

def avg(iterable):
    val = 0.
    for i, x in enumerate(iterable):
        val += (x - val)/(i+1)
    return val

def to_syms(R, *monoms):
    """
    Get the symbols of an ideal I
    """
    return [prod(R(R.symbols[i])**j
                for (i, j) in enumerate(monom))
                    for monom in monoms]

def smaller_elements_(lst, idx = 0, order=grevlex):
    """
    Returns all elements smaller than the item in lst
    """
    assert order == grevlex
    if not isinstance(lst, list): lst = list(lst)
    if idx == 0: yield tuple(lst)
    if idx == len(lst)-1:
        yield tuple(lst)
        return

    tmp, tmp_ = lst[idx], lst[idx+1]
    while lst[idx] > 0:
        lst[idx] -= 1
        lst[idx+1] += 1
        for elem in smaller_elements_(lst, idx+1, order): yield elem
    lst[idx], lst[idx+1] = tmp, tmp_

def smaller_elements(lst, grade=None, idx = 0, order=grevlex):
    if not isinstance(lst, list): lst = list(lst)
    while True:
        for elem in smaller_elements_(lst, 0, order): yield elem
        # Remove one from the largest element
        for i in range(len(lst)-1, -1, -1):
            if lst[i] > 0: lst[i] -= 1; break
        else: break

def dominated_elements(lst, idx = 0):
    """
    Iterates over all elements that are dominated by the input list.
    For example, (2,1) returns [(2,1), (2,0), (1,1), (1,0), (0,0), (0,1)]
    """
    # Stupid check
    if not isinstance(lst, list): lst = list(lst)
    if idx == len(lst): yield tuple(lst)
    else:
        tmp = lst[idx]
        # For each value of this index, update other values
        while lst[idx] >= 0:
            for elem in dominated_elements(lst, idx+1): yield elem
            lst[idx] -= 1
        lst[idx] = tmp

def test_dominated_elements():
    """Simple test of generating dominated elements"""
    L = list(dominated_elements((2,1)))
    assert L == [(2,1), (2,0), (1,1), (1,0), (0,1), (0,0)]

def support(fs, order=grevlex):
    """
    Get the terms spanned by support of
    f_1, ... f_n
    """
    O = set(chain.from_iterable(f.monoms() for f in fs))
    return sorted(O, key=order, reverse=True)

def order_ideal(fs, order=grevlex):
    """
    Return the order ideal spanned by these monomials.
    """
    O = set([])
    for t in support(fs, order):
        if t not in O:
            O.update(dominated_elements(list(t)))
    return sorted(O, key=grevlex, reverse=True)

def lt(arr, tau=0):
    """
    Get the leading term of arr > tau
    """
    if arr.ndim == 1:
        idxs, = arr.nonzero()
    elif arr.ndim == 2:
        assert arr.shape[0] == 1
        idxs = list(zip(*arr.nonzero()))
    else:
        raise Exception("Array of unexpected size: " + arr.ndim)
    for idx in idxs:
        elem = arr[idx]
        if abs(elem) > tau:
            if arr.ndim == 1:
                return idx, elem
            elif arr.ndim == 2:
                return idx[1], elem
    return 0, arr[0]

def lm(arr, tau=0):
    """Returns leading monomial"""
    return lt(arr, tau)[0]

def lc(arr, tau=0):
    """Returns leading coefficient"""
    return lt(arr, tau)[1]

def lt_normalize(R, tau=0):
    """
    Normalize to have the max term be 1
    """
    for i in range(R.shape[0]):
        R[i] /= lc(R[i], tau)
    return R

def row_normalize_leadingone(R, tau = eps):
    """
    Normalize rows to have leading ones
    """
    for r in R:
        lead = np.trim_zeros(r)[0]
        r /= lead
    return R

def row_normalize(R, tau = eps):
    """
    Normalize rows to have unit norm
    """
    for r in R:
        li = norm(r)
        if li < tau:
            r[:] = 0
        else:
            r /= li
    return R

def row_reduce(R, tau = eps):
    """
    Zero all rows with leading term from below
    """
    nrows, _ = R.shape
    for i in range(nrows-1, 0, -1):
        k, v = lt(R[i,:], tau)
        if v > tau:
            for j in range(i):
                R[j, :] -= R[i,:] * R[j,k] / R[i,k]
        else:
            R[i, :] = 0

    return R

def srref(A, tau=eps):
    """
    Compute the stabilized row reduced echelon form.
    """
    # TODO: Make sparse compatible
    if isinstance(A, sc.sparse.base.spmatrix):
        A = A.todense()
    A = array(A)
    m, n = A.shape

    Q = []
    R = zeros((min(m,n), n)) # Rows

    for i, ai in enumerate(A.T):
        # Remove any contribution from previous rows
        for j, qj in enumerate(Q):
            R[j, i] = ai.dot(qj)
            ai -= ai.dot(qj) * qj
        li = norm(ai)
        if li > tau:
            assert len(Q) < min(m,n)
            # Add a new column to Q
            Q.append(ai / li)
            # And write down the contribution
            R[len(Q)-1, i] = li

    # Convert to reduced row echelon form
    row_reduce(R, tau)

    # row_normalize
    row_normalize(R, tau)

    return array(Q).T, R

def test_srref():
    W = np.matrix([[  0.000e+00,   0.000e+00],
        [  0.000e+00,   0.000e+00],
        [  0.000e+00,   0.000e+00],
        [  5.020e-17,   1.180e-16],
        [ -4.908e-01,   6.525e-01],
        [ -8.105e-01,  -9.878e-02],
        [  0.000e+00,   0.000e+00],
        [  0.000e+00,   0.000e+00],
        [  3.197e-01,   7.513e-01]]).T
    return srref(W)

def simultaneously_diagonalize(Ms):
    """
    Simultaneously diagonalize a set of matrices.
    * Currently uses a crappy "diagonalize one and use for the rest"
      method.
    TODO: Just sum up the matrices randomly and do eig.
    """
    #ipdb.set_trace()
    #R, L, err = jacobi_angles(*Ms)
    #assert err < 1e-5
    #return L, R

    it = iter(Ms)
    M = next(it)
    l, R = eig(M)
    Ri = inv(R)
    L = [l]
    for M in it:
        l = diag(Ri.dot(M).dot(R))
        L.append(l)
    return L, R

def truncated_svd(M, epsilon=eps):
    """
    Computed the truncated version of M from SVD
    """
    U, S, V = svd(M)
    S = S[abs(S) > epsilon]
    return U[:, :len(S)], S, V[:len(S),:]

def closest_permuted_vector( a, b ):
    """Find a permutation of b that matches a most closely (i.e. min |A
    - B|_2)"""

    # The elements of a and b form a weighted bipartite graph. We need
    # to find their minimal matching.
    assert( a.shape == b.shape )
    n, = a.shape

    W = sc.zeros( (n, n) )
    for i in range( n ):
        for j in range( n ):
            W[i, j] = (a[i] - b[j])**2

    m = Munkres()
    matching = m.compute( W )
    matching.sort()
    _, bi = list(zip(*matching))
    return b[array(bi)]

def closest_permuted_matrix( A, B ):
    """Find a _row_ permutation of B that matches A most closely (i.e. min |A
    - B|_F)"""

    # The rows of A and B form a weighted bipartite graph. The weights
    # are computed using the vector_matching algorithm.
    # We need to find their minimal matching.
    assert( A.shape == B.shape )

    n, _ = A.shape
    m = Munkres()

    # Create the weight matrix
    W = sc.zeros( (n, n) )
    for i in range( n ):
        for j in range( n ):
            # Best matching between A and B
            W[i, j] = norm(A[i] - B[j])

    matching = m.compute( W )
    matching.sort()
    _, rowp = list(zip(*matching))
    rowp = array( rowp )
    # Permute the rows of B according to Bi
    B_ = B[ rowp ]

    return B_

def fix_parameters(true, guess, weights):
    """Find a column permutation of guess parameters that matches true parameters most closely (i.e. min |A
    - B|_F) also apply this to weights"""

    # The rows of A and B form a weighted bipartite graph. The weights
    # are computed using the vector_matching algorithm.
    # We need to find their minimal matching.
    assert true.shape == guess.shape

    d, k = true.shape
    m = Munkres()

    # Create the weight matrix
    W = sc.zeros((k, k))
    for i in range(k):
        for j in range(k):
            # Best matching between A and B
            W[i, j] = norm(true.T[i] - guess.T[j])

    matching = m.compute(W)
    matching.sort()
    _, colp = list(zip(*matching))
    colp = array(colp)
    # Permute the rows of B according to Bi
    guess = guess[:, colp]
    weights = weights[colp]

    return weights, guess


def save_example(R, I, out=sys.stdout):
    out.write(",".join(map(str, R.symbols)) +"\n")
    for i in I:
        out.write(str(i) + "\n")

def partitions(n, d):
    """
    Lists the partitions of d objects into n categories.
    partitions(3,2) = [(2,0,0), (1,1,0), (1,0,1), (
    """

    if n == 1:
        yield (d,)
    else:
        for i in range(d, -1, -1):
            for tup in partitions(n-1,d-i):
                yield (i,) + tup

def orthogonal(n):
    """Generate a random orthogonal 'd' dimensional matrix, using the
    the technique described in:
    Francesco Mezzadri, "How to generate random matrices from the
    classical compact groups"
    """
    n = int( n )
    z = sc.randn(n, n)
    q,r = sc.linalg.qr(z)
    d = sc.diagonal(r)
    ph = d/sc.absolute(d)
    q = sc.multiply(q, ph, q)
    return q

def hermite_coeffs(N=6):
    """
    helper function to generate coeffs of the Gaussian moments they are
    non-neg and equal in abs to the coeffs hermite polynomials which can
    be generated via a simple recurrence.
    For usage see test_1dmog of test_MomentMatrix
    """
    K = N
    A = np.zeros((N,K), dtype=np.int)
    # the recurrence formula to get coefficients of the hermite polynomails
    A[0,0] = 1; A[1,1] = 1; #A[2,0]=-1; A[2,2]=1;
    for n in range(1,N-1):
        for k in range(K):
            A[n+1,k] = -n*A[n-1,k] if k==0 else A[n,k-1] - n*A[n-1,k]
    return A

def chunked_update( fn, start, step, stop):
    """Run @fn with arguments @start to @stop in @step sized blocks."""

    iters = int( (stop - start)/step )
    for i in range( iters ):
        fn( start, start + step )
        start += step
    if start < stop:
        fn( start, stop )

def slog( x ):
    """Safe log - preserve 0"""
    if type(x) == sc.ndarray:
        y = sc.zeros( x.shape )
        y[ x > 0 ] = sc.log( x[ x > 0 ] )
    else:
        y = 0.0 if x == 0 else sc.log(x)

    return y


def permutation( n ):
    """Generate a random permutation as a sequence of swaps"""
    n = int( n )
    lst = sc.arange( n )
    sc.random.shuffle( lst )
    return lst

def wishart(n, V, nsamples=1):
    """wishart: Sample a matrix from a Wishart distribution given
    by a shape paramter n and a scale matrix V
    Based on: W. B. Smith and R. R. Hocking, Algorithm AS 53: Wishart
    Variate Generator, Applied Statistic, 21, 341

    Under the GPL License
    From the Astrometry project: http://astrometry.net/

    W(W|n,V) = |W|^([n-1-p]/2) exp(-Tr[V^(-1)W]/2)/ ( 2^(np/2) |V|^(n/2)
    pi^(p(p-1)/2) Prod_{j=1}^p \Gamma([n+1-j]/2) )
    where p is the dimension of V

    Input:
       n        - shape parameter (> p-1)
       V        - scale matrix
       nsamples - (optional) number of samples desired (if != 1 a list is returned)

    Output:
       a sample of the distribution

    Dependencies:
       scipy
       scipy.stats.chi2
       scipy.stats.norm
       scipy.linalg.cholesky
       math.sqrt

    History:
       2009-05-20 - Written Bovy (NYU)
    """
    #Check that n > p-1
    p = V.shape[0]
    if n < p-1:
        return -1
    #First lower Cholesky of V
    L = sc.linalg.cholesky(V,lower=True)
    if nsamples > 1:
        out = []
    for kk in range(nsamples):
        #Generate the lower triangular A such that a_ii = (\chi2_(n-i+2))^{1/2} and a_{ij} ~ N(0,1) for j < i (i 1-based)
        A = sc.zeros((p,p))
        for ii in range(p):
            A[ii,ii] = sc.sqrt(sc.stats.chi2.rvs(n-ii+2))
            for jj in range(ii):
                A[ii,jj] = sc.stats.norm.rvs()
        #Compute the sample X = L A A\T L\T
        thissample = sc.dot(L,A)
        thissample = sc.dot(thissample,thissample.transpose())
        if nsamples == 1:
            return thissample
        else:
            out.append(thissample)
    return out

def monomial(xs, betas):
    r"""
    Computes \prod_{i=1}^D x_i^{beta_i}
    """
    ret = 1.
    for x, beta in zip(xs, betas):
        if beta != 0:
            ret *= (x**beta)
    return ret

def project_nullspace(A, b, x, randomize = 0):
    """
    Project a vector x onto the space of y where Ay=b
    not efficient if you always project the same A and b
    """
    y0,__,__,__ = scipy.linalg.lstsq(A,b)
    xnorm = x - y0
    U,S,V = scipy.linalg.svd(A)
    rank = np.sum(S>eps)
    B = V[rank:, :]

    noise = 0
    if randomize>0:
        dist = norm(V[0:rank,:].dot(xnorm))
        noise = B.T.dot(np.random.randn(B.shape[0],1))
    #ipdb.set_trace()
    return B.T.dot(B.dot(xnorm)) + y0 + noise

def test_project_nullspace():
    A = np.array([[1,0,-1],[1,1,0]])
    # b = A [1 0 0], B = [1,-1,1], P = [1,1,0]
    b = np.array([1,1])[:,np.newaxis]
    print('should stay fixed')
    print(project_nullspace(A,b,np.array([2,-1,1])[:, np.newaxis]))
    print('should also go to [2,-1,1]')
    print(project_nullspace(A,b,np.array([3,0,1])[:, np.newaxis]))

def column_aerr( M, M_ ):
    return max( [norm( mu_mu_[0] - mu_mu_[1] ) for mu_mu_ in zip( M.T, M_.T
        )] )

def column_rerr( M, M_ ):
    return max( [norm( mu_mu_1[0] - mu_mu_1[1] )/norm( mu_mu_1[0] ) for mu_mu_1 in zip(
        M.T, M_.T )] )

def dict_diff(d1, d2, p = 2):
    """
    Compute the difference between dicts
    """
    keys = set.union(set(d1.keys()), set(d2.keys()))
    diff = 0.
    for key in keys:
        diff += abs(d1.get(key, 0.) - d2.get(key, 0.))**p
    return diff ** (1./p)



def dict_mono_to_ind(monolist):
    dict = {}
    for i,mono in enumerate(monolist):
        dict[mono]=i
    return dict

def extract_solutions_lasserre(MM, ys, Kmax=10, tol=1e-6, maxdeg = None):
    """
    extract solutions via (unstable) row reduction described by Lassarre and used in gloptipoly
    MM is a moment matrix, and ys are its completed values
    @params - Kmax: the maximum rank allowed to extract
    @params - tol: singular values less than this is truncated.
    @params - maxdeg: only extract from the top corner of the matrix up to maxdeg
    """
    M = MM.numeric_instance(ys, maxdeg = maxdeg)
    
    Us,Sigma,Vs=np.linalg.svd(M)
    #
    count = min(Kmax,sum(Sigma>tol))
    # now using Lassarre's notation in the extraction section of
    # "Moments, Positive Polynomials and their Applications"
    T,Ut = srref(Vs[0:count,:])
    
    if Sigma[count] <= tol:
        print('lost %.7f' % Sigma[count])
    # inplace!
    row_normalize_leadingone(Ut)

    couldbes = np.where(Ut>0.9)
    ind_leadones = np.zeros(Ut.shape[0], dtype=np.int)
    for j in reversed(list(range(len(couldbes[0])))):
        ind_leadones[couldbes[0][j]] = couldbes[1][j]

    basis = [MM.row_monos[i] for i in ind_leadones]
    dict_row_monos = dict_mono_to_ind(MM.row_monos)
    
    Ns = {}
    bl = len(basis)
    # create multiplication matrix for each variable
    for var in MM.vars:
        Nvar = np.zeros((bl,bl))
        for i,b in enumerate(basis):
            Nvar[:,i] = Ut[ :,dict_row_monos[var*b] ]
        Ns[var] = Nvar

    N = np.zeros((bl,bl))
    for var in Ns:
        N+=Ns[var]*np.random.randn()
    T,Q=scipy.linalg.schur(N)

    sols = {}

    quadf = lambda A, x : np.dot(x, np.dot(A,x))
    for var in MM.vars:
        sols[var] = np.array([quadf(Ns[var], Q[:,j]) for j in range(bl)])
    #ipdb.set_trace()
    return sols

def extract_solutions_lasserre_average(MM, ys, Kmax=10, tol=1e-6, numiter=10):
    """
    extract solutions via (unstable) row reduction described by Lassarre and used in gloptipoly
    MM is a moment matrix, and ys are its completed values
    """
    M = MM.numeric_instance(ys)
    Us,Sigma,Vs=np.linalg.svd(M)
    #
    count = min(Kmax,sum(Sigma>tol))
    # now using Lassarre's notation in the extraction section of
    # "Moments, Positive Polynomials and their Applications"
    sols = {}
    totalweight = 0;
    for i in range(numiter):
        
        T,Ut = srref(M[0:count,:])
        
        if Sigma[count] <= tol:
            print('lost %.7f' % Sigma[count])
        # inplace!
        row_normalize_leadingone(Ut)

        couldbes = np.where(Ut>0.9)
        ind_leadones = np.zeros(Ut.shape[0], dtype=np.int)
        for j in reversed(list(range(len(couldbes[0])))):
            ind_leadones[couldbes[0][j]] = couldbes[1][j]

        basis = [MM.row_monos[i] for i in ind_leadones]
        dict_row_monos = dict_mono_to_ind(MM.row_monos)

        Ns = {}
        bl = len(basis)
        # create multiplication matrix for each variable
        for var in MM.vars:
            Nvar = np.zeros((bl,bl))
            for i,b in enumerate(basis):
                Nvar[:,i] = Ut[ :,dict_row_monos[var*b] ]
            Ns[var] = Nvar

        N = np.zeros((bl,bl))
        for var in Ns:
            N+=Ns[var]*np.random.randn()
        T,Q=scipy.linalg.schur(N)

        quadf = lambda A, x : np.dot(x, np.dot(A,x))
        for var in MM.vars:
            sols[var] = np.array([quadf(Ns[var], Q[:,j]) for j in range(bl)])
            
    return sols

def extract_solutions_dreesen_proto(MM, ys, Kmax=10, tol=1e-5):
    """
    extract solutions dreesen's nullspace method
    this is the prototype implementation that does not match solutions!
    """
    M = MM.numeric_instance(ys)
    Us,Sigma,Vs=sc.linalg.svd(M)
    
    count = min(Kmax,sum(Sigma>tol))
    Z = Us[:,0:count]
    print('the next biggest eigenvalue we are losing is %f' % Sigma[count])

    dict_row_monos = dict_mono_to_ind(MM.row_monos)
    
    sols = {}
    for var in MM.vars:
        S1 = np.zeros( (len(MM.row_monos), len(MM.row_monos)) )
        Sg = np.zeros( (len(MM.row_monos), len(MM.row_monos)) )
        # a variable is in basis of the current var if var*basis in row_monos
        basis = []
        i = 0
        for mono in MM.row_monos:
            if mono*var in MM.row_monos:
                basis.append(mono)
                basisind = dict_row_monos[mono]
                gind = dict_row_monos[mono*var]
                
                S1[i, basisind] = 1
                Sg[i, gind] = 1
                i += 1
        S1 = S1[0:i,:]
        Sg = Sg[0:i,:]
        
        A = Sg.dot(Z)
        B = S1.dot(Z)
        
        # damn this, cant i just have a GE solver that works for non-square matrices?
        __,__,P = np.linalg.svd(np.random.randn(count,i), full_matrices = False)
        
        sols[var] = sc.real(sc.linalg.eigvals(P.dot(A),P.dot(B))).tolist()
        
    return sols

def extract_solutions_dreesen(MM, ys, Kmax=10, tol=1e-5):
    """
    extract solutions dreesen's nullspace method
    """
    M = MM.numeric_instance(ys)
    Us,Sigma,Vs=sc.linalg.svd(M)
    
    count = min(Kmax,sum(Sigma>tol))
    Z = Us[:,0:count]
    print('the next biggest eigenvalue we are losing is %f' % Sigma[count])

    dict_row_monos = dict_mono_to_ind(MM.row_monos)
    
    sols = {}
    
    S1list = []
    Sglist = []
    it = 0 # i total
    for var in MM.vars:
        S1 = np.zeros( (len(MM.row_monos), len(MM.row_monos)) )
        Sg = np.zeros( (len(MM.row_monos), len(MM.row_monos)) )
        # a variable is in basis of the current var if var*basis in row_monos
        basis = []
        i = 0
        for mono in MM.row_monos:
            if mono*var in MM.row_monos:
                basis.append(mono)
                basisind = dict_row_monos[mono]
                gind = dict_row_monos[mono*var]
                S1[i, basisind] = 1
                Sg[i, gind] = 1
                i += 1
        S1 = S1[0:i,:]
        Sg = Sg[0:i,:]
        S1list.append(S1)
        Sglist.append(Sg)

    S1s = np.zeros( (len(MM.row_monos)*len(MM.vars), len(MM.row_monos)) )
    Sgs = np.zeros( (len(MM.row_monos)*len(MM.vars), len(MM.row_monos)) )

    r = 0
    for i in range(len(S1list)):
        S1i = S1list[i]
        Sgi = Sglist[i]
        numrow = S1i.shape[0]
        S1s[r:r+numrow, :] = S1i
        Sgs[r:r+numrow, :] = Sgi
        r = r + numrow
                
    S1s = S1s[0:r,:]
    Sgs = Sgs[0:r,:]

    A = Sgs.dot(Z)
    B = S1s.dot(Z)
    
    __,__,P = np.linalg.svd(np.random.randn(count,r), full_matrices = False)
    Dproj, V = sc.linalg.eig(P.dot(A),P.dot(B))
    
    sols = {}
    for i,var in enumerate(MM.vars):
        Ai = Sglist[i].dot(Z)
        Bi = S1list[i].dot(Z)
        AiVBiV = sc.sum(Ai.dot(V) * Bi.dot(V), 0)
        BiVBiV = sc.sum(Bi.dot(V) * Bi.dot(V), 0)
        sols[var] = AiVBiV / BiVBiV
    #ipdb.set_trace()
    return sols

def test_solution_extractors():
    import sympy as sp
    from . import core as core
    x = sp.symbols('x')
    M = core.MomentMatrix(2, [x], morder='grevlex')
    ys = [1, 1.5, 2.5, 4.5, 8.5]
    sols_lass = extract_solutions_lasserre(M, ys)

    sols_dree = extract_solutions_dreesen(M, ys)
    print(sols_lass, sols_dree)
    print('true values are 1 and 2')
    
if __name__ == "__main__":
    test_solution_extractors()
