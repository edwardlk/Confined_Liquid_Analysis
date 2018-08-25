###############################################################################
#
# A program to analyze raw data to produce stiffness and damping curves as a
# function of distance.
#
# To do:
# - GUI interface?
# - clean up functions (i.e. convert Phase to rad)
# - Add list of necessary packages to README
# - % complete printouts
# - error handling for folder/file selection

import time
from os import path, listdir, makedirs
import numpy as np   #
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

from CLfuncs import Deflection, Movement, Amplitude, Phase, Stiffness, Damping
from CLfuncs import Relaxation, smooth, outputFiles, graphMax

csvHeader = (
    'Index,Distance(Ang),Tunnel(nA),Ipd(mV),Extin(V),SpareAdcCh1(V),'
    'SpareAdcCh2(V),RTunnel(nA),RIpd(mV),RExtin(V),RSpareAdcCh1(V),'
    'RSpareAdcCh2(V)')

# Designate input and output directories.
root = Tk()
root.withdraw()

info = ('Please select the folder that contains '
        'the data files you wish to analyze.')

srcDir = filedialog.askdirectory(parent=root, initialdir="/", title=info)
dstDir = path.join(srcDir, 'output')
csvDir = path.join(srcDir, 'csv')

# Get file list from source directory
dataFiles = listdir(srcDir)
dataFiles.sort()

info2 = 'Select the file that contains the constants.'
conLoc = filedialog.askopenfilename(parent=root,
                                    initialdir=srcDir, title=info2)
conFile = path.split(conLoc)[1]

start = time.time()

if conFile in dataFiles:
    dataFiles.remove(conFile)

# Make output directories if they do not exist
# if the directories do exist, deletes from dataFiles
if not path.exists(dstDir):
    makedirs(dstDir)
else:
    dataFiles.remove('output')
if not path.exists(csvDir):
    makedirs(csvDir)
else:
    dataFiles.remove('csv')

# Create output files' names
dataOutput = outputFiles(dataFiles, '-output.txt')
dataImg = outputFiles(dataFiles, '.png')
csvOutput = outputFiles(dataFiles, '.csv')

# Overall Constants
#   To get particular values, use constants[n-1,m]
#   n = file number
#   m = [index, slope, V_batt, sens, t_c, Phase_off, freq, stiff, vel, Temp]
constants = np.genfromtxt(conLoc, skip_header=1)

# TEST 06-25-2015: getting list of speeds from constant file
speeds = sorted(set(constants[:, 8]))

# Create CSVs
for x in range(len(dataFiles)):
    currentfile = dataFiles[x]
    outputfile = csvOutput[x]

    data = np.genfromtxt(path.join(srcDir, currentfile), skip_header=20,
                         skip_footer=1)
    np.savetxt(path.join(csvDir, outputfile), data, header=csvHeader,
               delimiter=',')

# Main analysis
for x in range(len(dataFiles)):
    currentfile = dataFiles[x]
    currentpic = dataImg[x]
    outputfile = dataOutput[x]
    data = np.genfromtxt(path.join(srcDir, currentfile), skip_header=20,
                         skip_footer=1)

    rows = data.shape[0]
    columns = data.shape[1]

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

    pos = np.zeros(rows)        # Actual Position (d+z)
    amp = np.zeros(rows)        # Amplitude
    phi = np.zeros(rows)        # Phase
    k_ts = np.zeros(rows)       # Interaction Stiffness
    gamma = np.zeros(rows)      # Damping Coefficient
    t_R = np.zeros(rows)        # Relaxation Time
    k_tsavg = np.zeros(rows)    # Interaction Stiffness

    for x2 in range(0, rows):
        phi[x2] = Phase(ADC1[x2])
        amp[x2] = Amplitude(Extin[x2], constants[x, 2], constants[x, 3],
                            constants[x, 1])

    phi0 = phi[0]  # changed to match shah analysis
    A0 = amp[8]
    Amax = max(amp[:99])

    for x3 in range(0, rows):
        pos[x3] = Deflection(Ipd[x3], constants[x, 3]) + Movement(x3)
        phi[x3] = Phase(ADC1[x3])
        k_ts[x3] = Stiffness(constants[x, 7], Amax, phi[x3], phi0, amp[x3])
        gamma[x3] = Damping(constants[x, 7], A0, phi[x3], phi0, amp[x3],
                            constants[x, 6])

    k_tsavg = smooth(k_ts, 11, 'hamming')
    gammaavg = smooth(gamma, 11, 'hamming')

    for x4 in range(0, rows):
        t_R[x4] = Relaxation(k_tsavg[x4], gammaavg[x4], constants[x, 6])

    # Output Calculations
    output = np.column_stack((Distance, pos, amp, phi, k_ts, k_tsavg, gamma,
                              gammaavg, t_R))

    np.savetxt(path.join(dstDir, outputfile), output,
               header='Distance Position Amplitude Phase Stiffness '
               'Stiffness_avg Damping Damping_avg Relaxation_Time',
               comments="")

    # # PLOT COMPARISON OF SMOOTHING METHODS
    # windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    # plot(Distance, k_ts)
    # for w in windows:
    #     plot(Distance, smooth(k_ts, 11, w))
    # l = ['original signal']
    # l.extend(windows)
    # legend(l, loc=2)
    # title("Smoothing k_ts")
    # show()
    # savefig(currentpic)

    # PLOT CALCULATED VALUES
    fig = plt.figure(figsize=(6, 7))

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
    ax3.set_ylim([0, graphMax(k_ts, 12)])
    for tl in ax3.get_yticklabels():
        tl.set_color('r')

    ax4 = ax3.twinx()
    ax4.plot(Distance, gammaavg, 'b.')
    ax4.set_ylabel('Damping Coefficient', color='b')
    ax4.set_ylim([0, graphMax(gammaavg, 0.002)])
    for tl in ax4.get_yticklabels():
        tl.set_color('b')

    ax5 = fig.add_subplot(313)
    ax5.plot(Distance, k_tsavg, 'r.')
    ax5.set_xlabel('Distance (Angstroms)')
    ax5.set_ylabel('Stiffness', color='r')
    ax5.set_ylim([0, graphMax(k_tsavg, 12)])
    for tl in ax5.get_yticklabels():
        tl.set_color('r')

    ax6 = ax5.twinx()
    ax6.plot(Distance, t_R, 'b.')
    ax6.set_ylabel('Relaxation Time', color='b')
    ax6.set_ylim([0, 0.010])
    for tl in ax6.get_yticklabels():
        tl.set_color('b')

    plt.subplots_adjust(left=0.1, right=0.85)
    plt.suptitle(
        r'Curve %d: %d $\AA$/s @ %.1f C' %
        (x+1, constants[x, 8], constants[x, 9]))

    plt.savefig(path.join(dstDir, currentpic))
    # plt.show()
    plt.close()

print('Finished analyzing', path.split(srcDir)[1])
print('It took {:.2f} seconds to analyze %d files.'.format(time.time()-start) %
      (len(dataFiles)))
