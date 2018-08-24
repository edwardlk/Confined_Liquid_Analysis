import math
import numpy as np


def Deflection(Ipd, Sensitivity):
    return (Ipd * 1000) * 10**(-10) / Sensitivity


def Movement(point):
    return (281.6 - 0.55 * point) * 10**(-10)


def Amplitude(ExtIn, V_cell, Sensitivity, Slope):
    return math.sqrt(2) * (V_cell - ExtIn) * (Sensitivity / 10) / Slope


def Phase(ADC1):
    return ADC1 * 180 / 9


def Stiffness(k_L, A_0, phi, phi_0, Amplitude):
    C1 = math.cos(math.pi * (phi - phi_0) / 180)
    return k_L * (A_0 * C1 / Amplitude - 1)


def Damping(k_L, A_0, phi, phi_0, Amplitude, frequency):
    S1 = math.sin(math.pi * (phi - phi_0) / 180)
    D1 = (Amplitude * 2 * math.pi * frequency)
    return -(k_L * A_0 * S1 / D1)


def Relaxation(k_ts, gamma, f):
    return k_ts / (gamma * (2 * math.pi * f)**2)


def smooth(x, window_len, window):
    if x.ndim != 1:
        raise ValueError('smooth only accepts 1-D array')
    if x.size < window_len:
        raise ValueError('input vector mut be larger than window size')
    if window_len < 3:
        return x
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError(
            "Window must be: 'flat', 'hanning', 'hamming','bartlett',"
            " or 'blackman'")

    end = window_len - int(window_len/2)

    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]

    if window == 'flat':    # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')

    return y[window_len-end:-window_len+end]


def outputFiles(dataFiles, addon):
    L = []
    for x in range(len(dataFiles)):
        temp = dataFiles[x]
        L.append(temp[:-4] + addon)
    return L


def graphMax(quantity, minMax):
    if np.amax(quantity) > minMax:
        return np.amax(quantity)
    else:
        return minMax
