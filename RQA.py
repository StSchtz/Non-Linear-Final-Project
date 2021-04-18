# -*- coding: utf-8 -*-

import numpy as np
from itertools import chain


def logistic_map(x0, r, T):
    """
    Generates a logistic map time series. Automatically discards transients (half the length of the time series).
    """
    #  Initialize the time series array
    timeSeries = np.empty(T + int(0.5*T))
    # iterate
    timeSeries[0] = x0
    for i in range(1, len(timeSeries)):
        xn = timeSeries[i-1]
        timeSeries[i] = r * xn * (1 - xn)
    # return without transients
    return timeSeries[int(0.5*T):]



def embed(x, m, tau):
    """
    Embeds a scalar time series in m dimensions with time delay tau.
    """
    n = len(x)
    k = n - (m - 1) * tau
    z = np.zeros((k, m), dtype="float")
    for i in range(k):
        z[i] = [x[i + j * tau] for j in range(m)]

    return z


def count_num_lines(arr):
    """returns a list of line lengths contained in given array."""
    line_lens = []
    counting = False
    l = 0
    for i in range(len(arr)):
        if counting:
            if arr[i] == 0:
                l += 1
                line_lens.append(l)
                l = 0
                counting = False
            elif arr[i] == 1:
                l += 1
                if i == len(arr) - 1:
                    l += 1
                    line_lens.append(l)
        elif not counting:
            if arr[i] == 1:
                counting = True
    return line_lens


def diagonal_lines_hist(R):
    """returns the histogram P(l) of diagonal lines of length l."""
    line_lengths = []
    for i in range(1, len(R)):
        d = np.diag(R, k=i)
        ll = count_num_lines(d)
        line_lengths.append(ll)
    line_lengths = np.array(list(chain.from_iterable(line_lengths)))
    bins = np.arange(0.5, line_lengths.max() + 0.1, 1.)
    num_lines, _ = np.histogram(line_lengths, bins=bins)
    return num_lines, bins, line_lengths



def det(R, lmin=None, hist=None):
    """returns DETERMINISM for given recurrence matrix R."""
    if not lmin:
        lmin = int(0.1 * len(R))
    if not hist:
        nlines, bins, ll = diagonal_lines_hist(R)
    else:
        nlines, bins, ll = hist[0], hist[1], hist[2]
    Pl = nlines.astype('float')
    l = (0.5 * (bins[:-1] + bins[1:])).astype('int')
    idx = l >= lmin
    num = l[idx] * Pl[idx]
    den = l * Pl
    DET = num.sum() / den.sum()
    return DET



def lacunarity(x, w, norm=False):
    """
    Returns (normalized) Lacunarity of a binary data matrix/image for a list of specified box widths w.
    """
    xc = 1-x # complementary image
    N = x.shape[0]
    l = np.zeros(len(w)) # lacunarity vector
    ln = np.zeros(len(w)) # normalized lacunarity vector

    # iterate over different box widths
    cnt = 0
    for v in w:

        # distribution of masses
        M = np.zeros(N * N);
        Mc = np.zeros(N * N);

        # move box through the image
        k = 0
        for i in np.arange(0,N-v,v):
            for j in np.arange(0,N-v,v):
               M[k] = sum(sum(x[i:i+v,j:j+v]));
               Mc[k] = sum(sum(xc[i:i+v,j:j+v]));
               k += 1

        # first and second moment
        m1 = np.mean(M[0:k-1]); m2 = np.mean(M[0:k-1]**2.); 
        m1c = np.mean(Mc[0:k-1]); m2c = np.mean(Mc[0:k-1]**2.); 

        # lacunarity
        l_o = m2/(m1**2.); 
        l[cnt] = l_o;
        l_c = m2c/(m1c**2.);
        ln[cnt] = 2. - 1./l_o - 1./l_c;
        cnt += 1;
    # return normalized or non-normalized Lacunarity
    if norm==True:
        return ln
    else:
        return l