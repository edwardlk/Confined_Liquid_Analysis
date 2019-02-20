import numpy as np
import pandas as pd
from openpyxl import load_workbook
from shutil import copyfile
from os import path, listdir, makedirs
from tkinter import Tk, filedialog

csvHeader2 = ('Index', 'Distance(Ang)', 'Tunnel(nA)', 'Ipd(mV)', 'Extin(V)',
              'ADC1(V)', 'ADC2(V)', 'ADC3(V)', 'ADC4(V)', 'ADC5(V)', 'ADC6(V)',
              'ADC7(V)', 'ADC8(V)', 'RTunnel(nA)', 'RIpd(mV)', 'RExtin(V)',
              'RADC1(V)', 'RADC2(V)', 'RADC3(V)', 'RADC4(V)', 'RADC5(V)',
              'RADC6(V)', 'RADC7(V)', 'RADC8(V)')

root = Tk()
root.withdraw()

info = ('Please select the folder that contains the data to convert to xlsx.')

srcDir = filedialog.askdirectory(parent=root, initialdir="/", title=info)
dstDir = path.join(srcDir, 'xlsx-output')

if not path.exists(dstDir):
    makedirs(dstDir)

info2 = 'Select the file that contains the constants.'
conLoc = filedialog.askopenfilename(parent=root, initialdir=srcDir,
                                    title=info2)
constants = pd.read_csv(conLoc, sep='\t', header=0)

# Get file list from source directory
dataFiles = listdir(srcDir)
dataFiles.sort()

for x in range(len(dataFiles)):
    dataFile = path.join(srcDir, dataFiles[x])
    data = np.genfromtxt(dataFile, skip_header=21, skip_footer=1)

    df = pd.DataFrame(data=data, columns=csvHeader2)

    # constOut = constants[x:(x+1)].transpose()
    constOut = constants.iloc[x].transpose()
    excelBookName = (str(constants.loc[x, 'Speed(A/s)']) + 'Aps-LTAFM-'
                     + str(x).zfill(3) + '.xlsx')
    excelBookLoc = path.join(dstDir, excelBookName)

    # Put xlsx model file in github folder?
    copyfile('___Aps-LTAFM-000v4.xlsx', excelBookLoc)

    book = load_workbook(excelBookLoc)
    writer = pd.ExcelWriter(excelBookLoc, engine='openpyxl', mode='a')

    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

    df.to_excel(writer, sheet_name='raw', index=False)
    constOut.to_excel(writer, sheet_name='Approach', header=False, startrow=1)

    writer.save()
