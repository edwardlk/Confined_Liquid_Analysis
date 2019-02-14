import math
import numpy as np
from scipy.optimize import curve_fit


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


# May be worth it just to drop null data from the start...
def joinAR(Extin, R_Extin, Distance):
    """Joins the approach and retract arrays into 1 array, after flipping the
    order of the retract array and rescaling the distance array so that the
    contact occurs at the origin. Returns the new arrays and the index of the
    first null point of the approach curve.
    """
    step = np.abs(Distance[0] - Distance[1])
    for x5 in range(len(Extin)):
        if Extin[x5] == 0 or x5 == len(Extin)-1:
            ExtinAR1 = np.zeros(x5)
            ExtinAR2 = np.zeros(x5+1)
            ExtinAR1 = Extin[:x5]
            DistAR1 = Distance[:x5] - Distance[x5] - step
            ExtinAR2 = R_Extin[:x5+1]
            DistAR2 = abs(Distance[:x5+1] - Distance[x5])
            ExtinAR = np.append(ExtinAR1, np.flip(ExtinAR2))
            DistAR = np.append(DistAR1, np.flip(DistAR2))
            break
    contactIndex = x5
    return ExtinAR, DistAR, contactIndex


def joinAR2(Extin, R_Extin, Distance, x5):
    step = np.abs(Distance[0] - Distance[1])
    ExtinAR1 = np.zeros(x5)
    ExtinAR2 = np.zeros(x5+1)
    ExtinAR1 = Extin[:x5]
    DistAR1 = Distance[:x5] - Distance[x5] - step
    ExtinAR2 = R_Extin[:x5+1]
    DistAR2 = abs(Distance[:x5+1] - Distance[x5])
    ExtinAR = np.append(ExtinAR1, np.flip(ExtinAR2))
    DistAR = np.append(DistAR1, np.flip(DistAR2))
    return ExtinAR, DistAR


def fit_sin(xx, yy):
    '''Fit sin to the input distance sequence, and return fitting parameters
        "amp", "wave_num", "phase", "offset", "wave_len", "vel" and
        "fitfunc"'''
    xx = np.array(xx)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(xx), (xx[1]-xx[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])  # excluding the 0 freq. "peak",
    guess_amp = np.std(yy) * 2.**0.5            # which is related to offset
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(x, A, k, p, c): return A * np.sin(k*x + p) + c
    popt, pcov = curve_fit(sinfunc, xx, yy, p0=guess)
    A, k, p, c = popt
    wav_len = (2.*np.pi) / k
    vel = wav_len * 0.4  # f = 0.4 Hz, change if different
    fitfunc = lambda x: A * np.sin(k*x + p) + c
    return {"amp": A, "wave_num": k, "phase": p, "offset": c,
            "wav_len": wav_len, "vel": vel, "fitfunc": fitfunc,
            "maxcov": np.max(pcov), "rawres": (guess, popt, pcov)}
# Example Usage
# dataLoc ='D:/ekram/Desktop/test_0.4Hz_30Aps.txt'
# data = np.genfromtxt(dataLoc, skip_header=1)
# rows = data.shape[0]
# columns = data.shape[1]
# (Index, Distance, Tunnel, Ipd, Extin, ADC1, ADC2, ADC3, ADC4, ADC5, ADC6, ADC7, ADC8) = data.T
# res = fit_sin(Distance, ADC3)
# print( "Amplitude=%(amp)s, Wave Number=%(wave_num)s, wavLen=%(wav_len)s, vel=%(vel)s, Max. Cov.=%(maxcov)s" % res )
# Dist2 = np.linspace(min(Distance), max(Distance), 5*len(Distance))
# # PLOT CALCULATED VALUES
# fig = plt.figure()
# plt.plot(Distance, ADC3, 'o')
# plt.plot(Dist2, res["fitfunc"](Dist2), "r-", label="y fit curve", linewidth=2)
# plt.show()
# plt.close('all')


def dac2temp(ADC4):
    # have lookup array here
    # find value closest to ADC[0]
    # slice lookup array around found value
    # lookup rest of values, return temperatures
    return 1
