# single analysis

from os import path  # , listdir, makedirs
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

from CLfuncs import Deflection, Movement, Amplitude, Phase, Stiffness, Damping
from CLfuncs import Relaxation, smooth, graphMax, joinAR, joinAR2

fileAsk = True
if fileAsk:
    # Designate data file to analyze.
    root = Tk()
    root.withdraw()
    info = 'Please select the data file you wish to analyze.'
    dataLoc = filedialog.askopenfilename(parent=root, initialdir="/",
                                         title=info)
    info2 = 'Select the file that contains the constants.'
    conLoc = filedialog.askopenfilename(parent=root, title=info2,
                                        initialdir=path.split(dataLoc)[0])
    print("dataLoc =", dataLoc)
    print("conLoc =", conLoc)
else:
    dataLoc = 'C:/Users/ekram/Desktop/2018-08-22_water/water-012.txt'
    conLoc = 'C:/Users/ekram/Desktop/2018-08-22_water/constants.txt'

dataFile = path.split(dataLoc)[1]
conFile = path.split(conLoc)[1]

# Obtain File Number
fileParts = dataFile[:-4].split("-")
fileNum = int(fileParts[len(fileParts)-1])

# Overall Constants
#   To get particular values, use constants[n-1,m]
#   n = file number
#   m = [index, slope, V_batt, sens, t_c, Phase_off, freq, stiff, vel, Temp]
constants = np.genfromtxt(conLoc, skip_header=1)

x = fileNum - 1

# # 0:Index 1:Dist(Ang) 2:Tunnel(nA) 3:Ipd(mV) 4:Extin(V) 5:ADC1(V) 6:ADC2(V)
# # 7:R_Tunnel(nA) 8:R_Ipd(mV) 9:R_Extin(V) 10:R_ADC1(V) 11:R_ADC2(V)
# data = np.genfromtxt(dataLoc, skip_header=20, skip_footer=1)
# rows = data.shape[0]
# columns = data.shape[1]
# (Index, Distance, Tunnel, Ipd, Extin, ADC1, ADC2, R_Tunnel, R_Ipd, R_Extin,
#     R_ADC1, R_ADC2) = data.T

# Updated for new electronics & software - 8 ADCs & need to flip the ExtIn
data = np.genfromtxt(dataLoc, skip_header=21, skip_footer=1)
rows = data.shape[0]
columns = data.shape[1]
(Index, Distance, Tunnel, Ipd, Extin, ADC1, ADC2, ADC3, ADC4, ADC5, ADC6,
    ADC7, ADC8, R_Tunnel, R_Ipd, R_Extin, R_ADC1, R_ADC2, R_ADC3, R_ADC4,
    R_ADC5, R_ADC6, R_ADC7, R_ADC8) = data.T
Extin = -Extin

pos = np.zeros(rows)        # Actual Position (d+z)
amp = np.zeros(rows)        # Amplitude
R_amp = np.zeros(rows)
phi = np.zeros(rows)        # Phase
R_phi = np.zeros(rows)
k_ts = np.zeros(rows)       # Interaction Stiffness
R_k_ts = np.zeros(rows)
gamma = np.zeros(rows)      # Damping Coefficient
R_gamma = np.zeros(rows)
t_R = np.zeros(rows)	    # Relaxation Time
R_t_R = np.zeros(rows)
k_tsavg = np.zeros(rows)    # Interaction Stiffness

for x2 in range(0, rows):
    phi[x2] = Phase(ADC1[x2])
    R_phi[x2] = Phase(R_ADC1[x2])
    amp[x2] = Amplitude(Extin[x2], constants[x, 2], constants[x, 3],
                        constants[x, 1])
    R_amp[x2] = Amplitude(R_Extin[x2], constants[x, 2], constants[x, 3],
                          constants[x, 1])

phi0 = phi[0]  # changed to match shah analysis
A0 = amp[8]
Amax = max(amp[:99])

for x3 in range(0, rows):
    pos[x3] = Deflection(Ipd[x3], constants[x, 3]) + Movement(x3)
    k_ts[x3] = Stiffness(constants[x, 7], Amax, phi[x3], phi0, amp[x3])
    R_k_ts[x3] = Stiffness(constants[x, 7], Amax, R_phi[x3], phi0, R_amp[x3])
    gamma[x3] = Damping(constants[x, 7], A0, phi[x3], phi0, amp[x3],
                        constants[x, 6])
    R_gamma[x3] = Damping(constants[x, 7], A0, R_phi[x3], phi0, R_amp[x3],
                          constants[x, 6])

k_tsavg = smooth(k_ts, 11, 'hamming')
R_k_tsavg = smooth(R_k_ts, 11, 'hamming')
gammaavg = smooth(gamma, 11, 'hamming')
R_gammaavg = smooth(R_gamma, 11, 'hamming')

for x4 in range(0, rows):
    t_R[x4] = Relaxation(k_tsavg[x4], gammaavg[x4], constants[x, 6])
    R_t_R[x4] = Relaxation(R_k_tsavg[x4], R_gammaavg[x4], constants[x, 6])

ExtinAR, DistAR, x5 = joinAR(Extin, R_Extin, Distance)
ADC1AR, Dist1AR = joinAR2(ADC1, R_ADC1, Distance, x5)
k_tsAR, Dist2AR = joinAR2(k_tsavg, R_k_tsavg, Distance, x5)
gammaavgAR, Dist3AR = joinAR2(gammaavg, R_gammaavg, Distance, x5)
t_RAR, Dist4AR = joinAR2(t_R, R_t_R, Distance, x5)

print(len(DistAR))
if len(DistAR) < 207:
    graphT = 0
    graphB = len(DistAR)
else:
    temp = int(len(DistAR)/2)
    graphT = temp - 103
    graphB = temp + 103

# PLOT CALCULATED VALUES
fig = plt.figure(figsize=(6, 7))
plt.subplots_adjust(top=0.93)

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
# ax3.set_xlabel('Distance (Angstroms)')
ax3.set_ylabel('Stiffness', color='r')
ax3.set_ylim([0, graphMax(k_tsAR, 12)])
for tl in ax3.get_yticklabels():
    tl.set_color('r')

ax4 = ax3.twinx()
ax4.plot(Dist3AR[graphT:graphB], gammaavgAR[graphT:graphB], 'b.-')
ax4.set_ylabel('Damping Coefficient', color='b')
# ax4.set_ylim([0, graphMax(gammaavg, 0.002)])
for tl in ax4.get_yticklabels():
    tl.set_color('b')

ax5 = fig.add_subplot(313)
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
    r'Curve %d: %d $\AA$/s @ %.1f C' %
    (x+1, constants[x, 8], constants[x, 9]))

plt.show()
plt.close('all')

print('Done')
