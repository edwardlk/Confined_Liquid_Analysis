import math
import time
import pandas as pd
import multiprocessing as mp
from openpyxl import load_workbook
from shutil import copyfile
from os import path, listdir, makedirs
from tkinter import Tk, filedialog

info = 'Please select the folder that contains the data to convert to xlsx.'
info2 = 'Select the file that contains the constants.'
csvHeader2 = ('Index', 'Distance(Ang)', 'Tunnel(nA)', 'Ipd(mV)', 'Extin(V)',
              'ADC1(V)', 'ADC2(V)', 'ADC3(V)', 'ADC4(V)', 'ADC5(V)', 'ADC6(V)',
              'ADC7(V)', 'ADC8(V)', 'RTunnel(nA)', 'RIpd(mV)', 'RExtin(V)',
              'RADC1(V)', 'RADC2(V)', 'RADC3(V)', 'RADC4(V)', 'RADC5(V)',
              'RADC6(V)', 'RADC7(V)', 'RADC8(V)')


def fileConvert(x, srcDir, dataFile, constants, dstDir):
    dataFilePath = path.join(srcDir, dataFile)
    df = pd.read_csv(dataFilePath)

    constOut = constants.iloc[x]

    excelBookName = (str(int(math.ceil(constants.at[x, 'Speed(A/s)'])))
                     .zfill(2) + 'Aps-LTAFM-' + str(x+1).zfill(3) + '.xlsx')
    excelBookLoc = path.join(dstDir, excelBookName)

    copyfile('D:/ekram/Desktop/__Aps-LTAFM-000v6.xlsx', excelBookLoc)

    book = load_workbook(excelBookLoc)
    writer = pd.ExcelWriter(excelBookLoc, engine='openpyxl', mode='a')

    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

    df.to_excel(writer, sheet_name='raw', index=False)
    constOut.to_excel(writer, sheet_name='Approach', header=False, startrow=1)

    writer.save()

    print('File %d completed.' % (x+1))


def main():
    root = Tk()
    root.withdraw()

    srcDir = filedialog.askdirectory(parent=root, initialdir="/", title=info)
    # srcDir = 'D:/ekram/Desktop/2019-03-17/csv'
    dstDir = path.join(srcDir, 'xlsx-output')

    # Get file list from source directory
    dataFiles = listdir(srcDir)
    dataFiles.sort()

    conLoc = filedialog.askopenfilename(parent=root, initialdir=srcDir,
                                        title=info2)
    # conLoc = 'D:/ekram/Desktop/2019-03-17/output/Speeds+Temps.csv'
    constants = pd.read_csv(conLoc, header=0)

    print(constants)

    if not path.exists(dstDir):
        makedirs(dstDir)
    else:
        dataFiles.remove('xlsx-output')

    start = time.time()

    pool = mp.Pool(processes=5)
    for x in range(len(dataFiles)):
        pool.apply_async(fileConvert, args=(x, srcDir, dataFiles[x],
                                            constants, dstDir,))
    pool.close()
    pool.join()

    print('Finished converting data.')
    print('It took {:.2f} seconds to convert %d files.'
          .format(time.time() - start) % (len(dataFiles)))


if __name__ == '__main__':
    main()
