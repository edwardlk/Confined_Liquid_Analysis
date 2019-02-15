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
import math
from os import path, listdir, makedirs
import numpy as np   #
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import scipy.stats as sps

from CLfuncs import Deflection, Movement, Amplitude, Phase, Stiffness, Damping
from CLfuncs import Relaxation, smooth, outputFiles, graphMax, joinAR, joinAR2
from CLfuncs import fit_sin, dac2temp

csvHeader = (
    'Index,Distance(Ang),Tunnel(nA),Ipd(mV),Extin(V),SpareAdcCh1(V),'
    'SpareAdcCh2(V),RTunnel(nA),RIpd(mV),RExtin(V),RSpareAdcCh1(V),'
    'RSpareAdcCh2(V)')

csvHeader2 = (
    'Index,Distance(Ang),Tunnel(nA),Ipd(mV),Extin(V),'
    'ADC1(V),ADC2(V),ADC3(V),ADC4(V),ADC5(V),ADC6(V),ADC7(V),ADC8(V),'
    'RTunnel(nA),RIpd(mV),RExtin(V),'
    'RADC1(V),RADC2(V),RADC3(V),RADC4(V),RADC5(V),RADC6(V),RADC7(V),RADC8(V),')

csvHeader3 = (
    'Index,Distance(Ang),Tunnel(nA),Ipd(mV),Extin(V),'
    'ADC1(V),ADC2(V),ADC3(V),ADC4(C),'
    'RTunnel(nA),RIpd(mV),RExtin(V),'
    'RADC1(V),RADC2(V),RADC3(V),RADC4(C),')

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
speed2 = np.zeros(len(dataFiles))
temps2 = np.zeros(len(dataFiles))

# Create CSVs
# for x in range(len(dataFiles)):
#     currentfile = dataFiles[x]
#     outputfile = csvOutput[x]
#
#     data = np.genfromtxt(path.join(srcDir, currentfile), skip_header=21,
#                          skip_footer=1)
#     for x1
# in range(data.shape[0]):
#         if (data[x1, 13] == data[x1, 14]) and (data[x1, 14] == data[x1, 15]):
#             data = data[:x1, :]
#             break
#     np.savetxt(path.join(csvDir, outputfile), data, header=csvHeader2,
#                delimiter=',')

# Main analysis
for x in range(len(dataFiles)):
    currentfile = dataFiles[x]
    currentpic = dataImg[x]
    outputfile = dataOutput[x]

    # skip_header = 21 for 1.16.13.3 8 ADC data, 20 for 4 ADC data
    data = np.genfromtxt(path.join(srcDir, currentfile), skip_header=21,
                         skip_footer=1)
    for x1 in range(data.shape[0]):
        if (data[x1, 13] == data[x1, 14]) and (data[x1, 14] == data[x1, 15]):
            data = data[:x1, :]
            break

    rows = data.shape[0]
    columns = data.shape[1]

    # Updated for new electronics & software - 8 ADCs & need to flip the ExtIn
    (Index, Distance, Tunnel, Ipd, Extin, ADC1, ADC2, ADC3, ADC4, ADC5, ADC6,
        ADC7, ADC8, R_Tunnel, R_Ipd, R_Extin, R_ADC1, R_ADC2, R_ADC3, R_ADC4,
        R_ADC5, R_ADC6, R_ADC7, R_ADC8) = data.T
    Extin = -Extin
    R_Extin = -R_Extin

    # Calc speed from ADC3, assumes f = 0.4 Hz, change in function if different
    res = fit_sin(Distance, ADC3)
    speed2[x] = res['vel']
    # conver to temperatures
    rtdT = dac2temp(ADC4)
    R_rtdT = dac2temp(R_ADC4)

    pos = np.zeros(rows)        # Actual Position (d+z)
    amp = np.zeros(rows)        # Amplitude
    phi = np.zeros(rows)        # Phase
    k_ts = np.zeros(rows)       # Interaction Stiffness
    gamma = np.zeros(rows)      # Damping Coefficient
    t_R = np.zeros(rows)        # Relaxation Time
    k_tsavg = np.zeros(rows)    # Interaction Stiffness
    R_amp = np.zeros(rows)        # reverse Amplitude
    R_phi = np.zeros(rows)        # reverse Phase
    R_k_ts = np.zeros(rows)       # reverse Interaction Stiffness
    R_gamma = np.zeros(rows)      # reverse Damping Coefficient
    R_t_R = np.zeros(rows)        # reverse Relaxation Time
    R_k_tsavg = np.zeros(rows)    # reverse Interaction Stiffness

    for x2 in range(0, rows):
        phi[x2] = Phase(ADC1[x2])
        amp[x2] = Amplitude(Extin[x2], constants[x, 2], constants[x, 3],
                            constants[x, 1])
        R_phi[x2] = Phase(R_ADC1[x2])
        R_amp[x2] = Amplitude(R_Extin[x2], constants[x, 2], constants[x, 3],
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
        R_k_ts[x3] = Stiffness(constants[x, 7], Amax, R_phi[x3], phi0,
                               R_amp[x3])
        R_gamma[x3] = Damping(constants[x, 7], A0, R_phi[x3], phi0, R_amp[x3],
                              constants[x, 6])

    k_tsavg = smooth(k_ts, 5, 'hamming')
    gammaavg = smooth(gamma, 5, 'hamming')
    R_k_tsavg = smooth(R_k_ts, 5, 'hamming')
    R_gammaavg = smooth(R_gamma, 5, 'hamming')

    for x4 in range(0, rows):
        t_R[x4] = Relaxation(k_tsavg[x4], gammaavg[x4], constants[x, 6])
        R_t_R[x4] = Relaxation(R_k_tsavg[x4], R_gammaavg[x4], constants[x, 6])

    ExtinAR, DistAR, contIndex = joinAR(Extin, R_Extin, Distance)
    ADC1AR, Dist1AR = joinAR2(ADC1, R_ADC1, Distance, contIndex)
    k_tsAR, Dist2AR = joinAR2(k_tsavg, R_k_tsavg, Distance, contIndex)
    gammaavgAR, Dist3AR = joinAR2(gammaavg, R_gammaavg, Distance, contIndex)
    t_RAR, Dist4AR = joinAR2(t_R, R_t_R, Distance, contIndex)

    # Get temperature at contact point
    contTemp = sps.mode(rtdT)[0][0]
    temps2[x] = contTemp

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

    if len(DistAR) < 207:
        graphT = 0
        graphB = len(DistAR)
    else:
        temp = int(len(DistAR)/2)
        graphT = temp - 103
        graphB = temp + 103

    # PLOT CALCULATED VALUES
    fig = plt.figure(figsize=(6, 7))

    ax1 = fig.add_subplot(311)
    plt.axvline(x=0)
    ax1.plot(DistAR[graphT:graphB], ExtinAR[graphT:graphB], 'r.-')
    # ax1.set_xlabel('Distance (Angstroms)')
    ax1.set_ylabel('Extin (V)', color='r')
    for tl in ax1.get_yticklabels():
        tl.set_color('r')

    ax2 = ax1.twinx()
    ax2.plot(Dist1AR[graphT:graphB], ADC1AR[graphT:graphB], 'b.-')
    ax2.set_ylabel('ADC Ch 2 (V)', color='b')
    for tl in ax2.get_yticklabels():
        tl.set_color('b')

    ax3 = fig.add_subplot(312)
    plt.axvline(x=0)
    ax3.plot(Dist2AR[graphT:graphB], k_tsAR[graphT:graphB], 'r.-')
    ax3.set_xlabel('Distance (Angstroms)')
    ax3.set_ylabel('Stiffness', color='r')
    # ax3.set_ylim([0, graphMax(k_tsAR, 12)])
    for tl in ax3.get_yticklabels():
        tl.set_color('r')

    ax4 = ax3.twinx()
    ax4.plot(Dist3AR[graphT:graphB], gammaavgAR[graphT:graphB], 'b.-')
    ax4.set_ylabel('Damping Coefficient', color='b')
    # ax4.set_ylim([0, graphMax(gammaavg, 0.002)])
    for tl in ax4.get_yticklabels():
        tl.set_color('b')

    ax5 = fig.add_subplot(313)
    plt.axvline(x=0)
    ax5.plot(Dist2AR[graphT:graphB], k_tsAR[graphT:graphB], 'r.-')
    ax5.set_xlabel('Distance (Angstroms)')
    ax5.set_ylabel('Stiffness', color='r')
    ax5.set_ylim([0, graphMax(k_tsAR, 12)])
    for tl in ax5.get_yticklabels():
        tl.set_color('r')

    ax6 = ax5.twinx()
    ax6.plot(Dist4AR[graphT:graphB], t_RAR[graphT:graphB], 'b.-')
    ax6.set_ylabel('Relaxation Time', color='b')
    ax6.set_ylim([-0.010, 0.010])
    for tl in ax6.get_yticklabels():
        tl.set_color('b')

    plt.subplots_adjust(left=0.1, right=0.85)
    plt.suptitle(
        r'Curve %d: %d $\AA$/s (%.2f measured) @ %.1f C' %
        (x+1, math.ceil(speed2[x]), speed2[x], contTemp))

    plt.savefig(path.join(dstDir, currentpic))
    # plt.show()
    plt.close()

    dataOut = np.column_stack((Index, Distance, Tunnel, Ipd, Extin, ADC1, ADC2,
                               ADC3, rtdT, R_Tunnel, R_Ipd, R_Extin, R_ADC1,
                               R_ADC2, R_ADC3, R_rtdT))

    np.savetxt(path.join(csvDir, csvOutput[x]), dataOut, header=csvHeader3,
               delimiter=',')

    print('File %d of %d completed.' % (x+1, len(dataFiles)))

output2 = np.column_stack((speed2, temps2))
np.savetxt(path.join(dstDir, 'Speeds+Temps.csv'), output2, delimiter=',',
           header='Calc_V,Temp(C)', comments="")

print('Finished analyzing', path.split(srcDir)[1])
print('It took {:.2f} seconds to analyze %d files.'.format(time.time()-start) %
      (len(dataFiles)))
