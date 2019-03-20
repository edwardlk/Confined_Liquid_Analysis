import math
import time
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

# Get file list from source directory
dataFiles = listdir(srcDir)
dataFiles.sort()

info2 = 'Select the file that contains the constants.'
conLoc = filedialog.askopenfilename(parent=root, initialdir=srcDir,
                                    title=info2)
constants = pd.read_csv(conLoc, header=0)

print(constants)

if not path.exists(dstDir):
    makedirs(dstDir)

start = time.time()

for x in range(len(dataFiles)):
    dataFile = path.join(srcDir, dataFiles[x])
    # data = np.genfromtxt(dataFile, skip_header=21, skip_footer=1)

    df = pd.read_csv(dataFile)

    # df = pd.DataFrame(data=data, columns=csvHeader2)

    # constOut = constants[x:(x+1)].transpose()
    constOut = constants.iloc[x]

    excelBookName = (str(int(math.ceil(constants.at[x, 'Speed(A/s)'])))
                     .zfill(2) + 'Aps-LTAFM-' + str(x+1).zfill(3) + '.xlsx')
    excelBookLoc = path.join(dstDir, excelBookName)

    # Put xlsx model file in github folder?
    copyfile('J:/_data/__Aps-LTAFM-000v6.xlsx', excelBookLoc)

    book = load_workbook(excelBookLoc)
    writer = pd.ExcelWriter(excelBookLoc, engine='openpyxl', mode='a')

    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

    df.to_excel(writer, sheet_name='raw', index=False)
    constOut.to_excel(writer, sheet_name='Approach', header=False, startrow=1)

    writer.save()

    print('File %d of %d completed.' % (x+1, len(dataFiles)))

print('Finished converting data.')
print('It took {:.2f} seconds to convert %d files.'.format(time.time()-start) %
      (len(dataFiles)))
