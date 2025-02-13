import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import os
import math
import numpy.polynomial.polynomial as poly
import sys
import argparse

def calc_rms(x, scale, i=1):
    """
    Root Mean Square in windows with linear detrending.

    Args:
    -----
      *x* : numpy.array
        one dimensional data vector
      *scale* : int
        length of the window in which RMS will be calculaed
    Returns:
    --------
      *rms* : numpy.array
        RMS data in each window with length len(x)//scale
    """
    # making an array with data divided in windows
    shape = (x.shape[0] // scale, scale)
    X = np.lib.stride_tricks.as_strided(x, shape=shape)
    # vector of x-axis points to regression
    scale_ax = np.arange(scale)
    rms = np.zeros(X.shape[0])
    for e, xcut in enumerate(X):
        coefs = poly.polyfit(scale_ax, xcut, i)
        xfit = poly.polyval(scale_ax, coefs)
        # detrending and computing RMS of each window
        rms[e] = np.sqrt(np.mean((xcut - xfit) ** 2))
    return rms


def dfa(x, scale_lim=[5, 13], scale_dens=0.5, i=2, saveAsPng=None, show=False, two_way=False):
    """
    Detrended Fluctuation Analysis - algorithm with measures power law
    scaling of the given signal *x*.
    More details about algorithm can be found e.g. here:
    Hardstone, R. et al. Detrended fluctuation analysis: A scale-free
    view on neuronal oscillations, (2012).

    Args:
    -----
      *x* : numpy.array
        one dimensional data vector
      *scale_lim* = [5,9] : list of lenght 2
        boundaries of the scale where scale means windows in which RMS
        is calculated. Numbers from list are indexes of 2 to the power
        of range.
      *scale_dens* = 0.25 : float
        density of scale divisions
      *show* = False
        if True it shows matplotlib picture
    Returns:
    --------
      *scales* : numpy.array
        vector of scales
      *fluct* : numpy.array
        fluctuation function
      *alpha* : float
        DFA exponent
    """
    # cumulative sum of data with substracted offset
    if two_way:
        rev_x = x[::-1]
        x = np.append(x, rev_x)
    y = np.cumsum(x - np.mean(x))
    # scales = (2 ** np.arange(scale_lim[0], scale_lim[1], scale_dens)).astype(np.int)
    scales = (2 ** np.arange(scale_lim[0], scale_lim[1], scale_dens)).astype(int)
    fluct = np.zeros(len(scales))
    # computing RMS for each window
    print("Started analyzing windows from size 2^" + str(scale_lim[0]) + " to size 2^"+str(scale_lim[1])+" with exponential step "+str(scale_dens))
    for e, sc in enumerate(scales):
        print("win size "+str(sc)+" - done!")
        fluct[e] = np.mean(np.sqrt(calc_rms(y, sc, i=i) ** 2))
    # fitting a line to rms data
    coeff = poly.polyfit(np.log2(scales), np.log2(fluct), 1)
    # coeff = np.polyfit(np.log2(scales), np.log2(fluct), 1)
    if show or saveAsPng is not None:
        fluctfit = 2 ** poly.polyval(np.log2(scales), coeff)
        plt.loglog(scales, fluct, 'bo')
        plt.loglog(scales, fluctfit, 'r', label=r'$\alpha$ = %0.2f' % coeff[-1])
        plt.title('DFA')
        plt.xlabel(r'$\log_{10}$(time window)')
        plt.ylabel(r'$\log_{10}$<F(t)>')
        plt.legend()
        if saveAsPng is not None:
            plt.savefig(saveAsPng)
            plt.close()
        else:
            plt.show()
    return scales, fluct, coeff[-1]


def scan_all(path, two_way=False, separator=',', scale_den=0.5, start_size=-1, end_size=-1, order=1):
    files = os.listdir(path).copy()
    for f in files:
        file = open(path+os.path.sep+f, "r", encoding='utf-8')
        t = file.read().lower()
        x = [int(v) for v in t.split(separator) if v != '']
        size = len(x)
        max_win = int(size/10) if end_size < 0 else end_size
        max_win_pow = int(math.log(max_win,2))
        if max_win_pow > 13:
            max_win_pow = 13;
        min_win = start_size if start_size > 0 else 10
        min_win_pow = int(math.log(min_win,2))
        base = os.path.basename(f)
        splitted_base = os.path.splitext(base)
        d = path
        for i in range(order, order+1):
            new_d = result_dir(d, order)
            
            if not os.path.exists(new_d):
                os.makedirs(new_d)

            image_path = new_d + os.path.sep + splitted_base[0] + "_dfa(" + str(i) + ")" + "_image.png"
            scales, fluct, alpha = dfa(x, i=i, scale_lim=[min_win_pow, max_win_pow], scale_dens=scale_den, saveAsPng=image_path, two_way=two_way)
            res = 'window size, fluct, log(size), log(fluct)'
            for e in range(len(scales)):
                res += '\n' +str(scales[e])+','+str(fluct[e])+','+str(np.log(scales[e]))+','+str(np.log(fluct[e]))
            res = res+'\n'+'alpha:,'+str(alpha)
            print("Result : "+str(alpha))
            new_file_path = new_d + os.path.sep + splitted_base[0] + "_dfa("+str(i)+")" + splitted_base[1]
            f = open(new_file_path, "w+", encoding='utf-8')
            f.write("".join(res))
            f.close()


def parse_args():
    arg = "-i 'D:\Python Projects\DFA_witn_env\data' --s1 100 --s2 1000 --exp 0.25 --sep , --tw"
    parser = argparse.ArgumentParser(description='Calculate DFA for input csv file')
    parser.add_argument('-i', dest='path', help='Path to the directory that contains target csv files', required=True)
    parser.add_argument('--s1', dest='start_size', help='Define preferred size of window to start DFA', type=int, default=-1,
                        required=False)
    parser.add_argument('--s2', dest='end_size',
                        help='Define preferred size of window when DFA should stop. Default = 1/10 of the whole size',
                        type=int,
                        default=-1,
                        required=False)
    parser.add_argument('--exp', dest='exp_delta', help='Step, to increase exponent of window size', type=float,
                        required=False, default=0.5)
    parser.add_argument('--sep', dest='separator',
                        help='Predefined separator of the values contained in the csv files.', required=False,
                        default=',')
    parser.add_argument('--tw', dest='two_way',
                        help='Should the two way algorithm be applied', required=False, type=bool, const=True,
                        default=False, nargs='?')
    parser.add_argument('--order', dest='order',
                        help='The order of DFA method (DFA0, DFA1, DFA2)', required=False, type=int,
                        default=1)

    args = parser.parse_args()
    scan_all(args.path, two_way=args.two_way, separator=args.separator, scale_den=args.exp_delta, start_size=args.start_size, end_size=args.end_size, order=args.order)
    print("Done! Check your result directory for details : "+str(result_dir(args.path, args.order)))


def result_dir(path, order):
    return str(path)+"_result_dfa"+str(order)


if __name__=='__main__':
    # read_args()
    parse_args()
