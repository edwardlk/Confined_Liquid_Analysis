################################################################################
##
## A program to analyze raw data to produce stiffness and damping curves as a
## function of distance.
##
## To do:
## - GUI interface?
## - clean up functions (i.e. convert Phase to rad)
## - Add list of necessary packages to README
## - % complete printouts
## - error handling for folder/file selection

import time

from os import *
import math
import numpy as np   #
from numpy import *  # Need to try to condense these two
from pylab import *
import matplotlib.pyplot as plt
import Tkinter, tkFileDialog

# Functions

def Deflection(Ipd, Sensitivity):
    return (Ipd * 1000) * 10**(-10) / Sensitivity

def Movement(point):
    return (281.6 - 0.55 * point) * 10**(-10)

def Amplitude(ExtIn, V_cell, Sensitivity, Slope):
    return math.sqrt(2) * (V_cell - ExtIn) * (Sensitivity / 10) / Slope

def Phase(ADC1):
    return ADC1 * 180 / 9

def Stiffness(k_L, A_0, phi, phi_0, Amplitude):
    return k_L * (A_0 * math.cos(math.pi * (phi - phi_0) / 180) /
                  Amplitude - 1)

def Damping(k_L, A_0, phi, phi_0, Amplitude, frequency):
    return -(k_L * A_0 * math.sin(math.pi * (phi - phi_0) / 180) /
             (Amplitude * 2 * math.pi * frequency))

def Relaxation(k_ts, gamma, f):
	return k_ts / (gamma * (2 * math.pi * f)**2)

def smooth(x,window_len,window):

    if x.ndim != 1:
        raise ValueError('smooth only accepts 1-D array')
    if x.size < window_len:
        raise ValueError('input vectoir mut be larger than window size')
    if window_len < 3:
        return quantity
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window must be: 'flat', 'hanning', 'hamming',\
                         'bartlett', or 'blackman'")

    end = window_len - int(window_len)/2

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    return y[window_len-end:-window_len+end]

def outputFiles(dataFiles, addon):
    L = []
    for x in range(len(dataFiles)):
        temp = dataFiles[x]
        L.append(temp[:-4] + addon)
    return L

def graphMax(quantity,minMax):
    if np.amax(quantity) > minMax:
        return np.amax(quantity)
    else:
        return minMax

## Designate input and output directories.

root = Tkinter.Tk()
root.withdraw()

info = 'Please select the folder that contains \
the data files you wish to analyze.'

srcDir = tkFileDialog.askdirectory(parent=root, initialdir="/", title=info)
dstDir = path.join(srcDir, 'output')

## Get file list from source directory

dataFiles = listdir(srcDir)
dataFiles.sort()

info2 = 'Select the file that contains the constants.'
conLoc = tkFileDialog.askopenfilename(parent=root,initialdir=srcDir,title=info2)
conFile = path.split(conLoc)[1]

start = time.time()

if conFile in dataFiles:
    dataFiles.remove(conFile)

##Make output directory if it does not exist
##if the directory does exist, deletes 'output' from dataFiles
if not path.exists(dstDir):
    makedirs(dstDir)
else:
    dataFiles.remove('output')

##Create output files' names
dataOutput = outputFiles(dataFiles, '-output.txt')
dataImg = outputFiles(dataFiles, '.png')

# Overall Constants
#   To get particular values, use constants[n-1,m]
#   n = file number
#   m = [index, slope, V_batt, sens, Amp, Phase, freq, stiff, vel]

constants = genfromtxt(conLoc, skip_header=1)

## TEST 06-25-2015: getting list of speeds from constant file
## working

speeds = sorted(set(constants[:,8]))

print speeds

## End TEST

for x in range(len(dataFiles)):
    currentfile = dataFiles[x]
    currentpic  = dataImg[x]
    outputfile  = dataOutput[x]

    data = genfromtxt(path.join(srcDir,currentfile), skip_header=20,
                      skip_footer=1)

    rows = data.shape[0]
    columns = data.shape[1]
                                # These will become:
    Index = np.zeros(rows)      # Index
    Distance = np.zeros(rows)   # Distance
    Ipd = np.zeros(rows)        # Photo Diode Current
    Extin = np.zeros(rows)      # External Input
    ADC1 = np.zeros(rows)       # Spare ADC Channel 1
    ADC2 = np.zeros(rows)       # Spare ADC Channel 2

    for x1 in range(0, rows):
        Index[x1] = data[x1, 0]
        Distance[x1] = data[x1, 1]
        Ipd[x1] = data[x1, 3]
        Extin[x1] = data[x1, 4]
        ADC1[x1] = data[x1, 5]
        ADC2[x1] = data[x1, 6]
                                # These will become:
    pos = np.zeros(rows)        # Actual Position (d+z)
    amp = np.zeros(rows)        # Amplitude
    phi = np.zeros(rows)        # Phase
    k_ts = np.zeros(rows)       # Interaction Stiffness
    gamma = np.zeros(rows)      # Damping Coefficient
    t_R = np.zeros(rows)	# Relaxation Time

    k_tsavg = np.zeros(rows)    # Interaction Stiffness

    for x2 in range(0, rows):
        phi[x2] = Phase(ADC1[x2])
        amp[x2] = Amplitude(Extin[x2], constants[x,2], constants[x,3],
                            constants[x,1])

    phi0 = phi[0]  ## changed to match shah analysis
    A0 = amp[8]
    Amax = max(amp[:99])

    for x3 in range(0, rows):
        pos[x3] = Deflection(Ipd[x3], constants[x,3]) + Movement(x3)
        phi[x3] = Phase(ADC1[x3])
        k_ts[x3] = Stiffness(constants[x,7], Amax, phi[x3], phi0, amp[x3])
        gamma[x3] = Damping(constants[x,7], A0, phi[x3], phi0, amp[x3],
                            constants[x,6])

    k_tsavg = smooth(k_ts,11,'hamming')
    gammaavg = smooth(gamma,11,'hamming')

    for x4 in range(0, rows):
        t_R[x4] = Relaxation(k_tsavg[x4], gammaavg[x4], constants[x,6])

    #Output Calculations
    output = np.column_stack((Distance, pos, amp, phi, k_ts, k_tsavg, gamma,
                              gammaavg, t_R))

    np.savetxt(path.join(dstDir,outputfile), output, header="Distance Position \
			   Amplitude Phase Stiffness Stiffness_avg Damping Damping_avg \
			   Relaxation_Time", comments="")

##    # PLOT COMPARISON OF SMOOTHING METHODS
##
##    windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
##
##    plot(Distance, k_ts)
##    for w in windows:
##            plot(Distance, smooth(k_ts,11,w))
##
##    l=['original signal']
##    l.extend(windows)
##
##    legend(l, loc=2)
##    title("Smoothing k_ts")
##    show()
    ##savefig(currentpic)

    # PLOT CALCULATED VALUES

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.plot(Distance, Extin, 'r.')
    # ax1.set_xlabel('Distance (Angstroms)')
    ax1.set_ylabel('Extin (V)', color='r')
    for tl in ax1.get_yticklabels():
        tl.set_color('r')

    ax2 = ax1.twinx()
    ax2.plot(Distance, ADC1, 'b.')
    ax2.set_ylabel('ADC Ch 2 (V)', color='b')
    for tl in ax2.get_yticklabels():
        tl.set_color('b')

    ax3 = fig.add_subplot(312)
    ax3.plot(Distance, k_tsavg, 'r.')
    ax3.set_xlabel('Distance (Angstroms)')
    ax3.set_ylabel('Stiffness', color='r')
    ax3.set_ylim([0, graphMax(k_ts,12)])
    for tl in ax3.get_yticklabels():
        tl.set_color('r')

    ax4 = ax3.twinx()
    ax4.plot(Distance, gammaavg, 'b.')
    ax4.set_ylabel('Damping Coefficient', color='b')
    ax4.set_ylim([0, graphMax(gammaavg,0.002)])
    for tl in ax4.get_yticklabels():
        tl.set_color('b')

    ax5 = fig.add_subplot(313)
    ax5.plot(Distance, k_tsavg, 'r.')
    ax5.set_xlabel('Distance (Angstroms)')
    ax5.set_ylabel('Stiffness', color='r')
    ax5.set_ylim([0, graphMax(k_tsavg,12)])
    for tl in ax5.get_yticklabels():
        tl.set_color('r')

    ax6 = ax5.twinx()
    ax6.plot(Distance, t_R, 'b.')
    ax6.set_ylabel('Relaxation Time', color='b')
    ax6.set_ylim([0, 0.010])
    for tl in ax6.get_yticklabels():
        tl.set_color('b')

    plt.subplots_adjust(left = 0.1, right = 0.85)
    plt.suptitle("Curve %d @ %d $\AA$/s" % (x+1,constants[x,8]))

    plt.savefig(path.join(dstDir,currentpic))
    ##plt.show()

    plt.close()

print "Finished analyzing", path.split(srcDir)[1]
print 'It took {:.2f} seconds to analyze %d files.'.format(time.time()-start) % (len(dataFiles))
