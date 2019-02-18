import numpy as np
import pandas as pd
from openpyxl import load_workbook
from shutil import copyfile

dataFile = 'D:/ekram/Desktop/2019-02-13/009.txt'
data = np.genfromtxt(dataFile, skip_header=21, skip_footer=1)

csvHeader2 = (
    'Index','Distance(Ang)','Tunnel(nA)','Ipd(mV)','Extin(V)','ADC1(V)',
    'ADC2(V)','ADC3(V)','ADC4(V)','ADC5(V)','ADC6(V)','ADC7(V)','ADC8(V)',
    'RTunnel(nA)','RIpd(mV)','RExtin(V)','RADC1(V)','RADC2(V)','RADC3(V)',
    'RADC4(V)','RADC5(V)','RADC6(V)','RADC7(V)','RADC8(V)')

df = pd.DataFrame(data=data, columns=csvHeader2)

constants = pd.read_csv('D:/ekram/Desktop/2019-02-13/constants.txt', sep='\t',
                        header=0)

constOut = constants[9:10].transpose()

print(constants)
print(constants[2:3])

excelBookName = (str(constants.loc[(9-1), 'Speed(A/s)']) + 'Aps-LTAFM-'
                 + str('009.xlsx'))

copyfile('___Aps-LTAFM-000v4.xlsx', excelBookName)

book = load_workbook(excelBookName)
writer = pd.ExcelWriter(excelBookName, engine='openpyxl', mode='a')

writer.book = book
writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

df.to_excel(writer, sheet_name='raw', index=False)
constOut.to_excel(writer, sheet_name='Approach', header=False, startrow=1)

writer.save()
