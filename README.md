A set of python script used for the analysis of small amplitude AFM force curves.
Data captured with a Nanomagnetics SPM controller with 8 ADC aux ports.
*List ADC inputs.*
A file containing the *constants* [e.g. the interferometer's slope, the oscillation frequency, etc.] need to be recorded and saved with the data.

### General Usage Procedure
1. All data files should be placed into an empty folder.
The constants sheet should be placed in there as well.
2. Run *analysis.py*. First it will ask you to select the folder that contains all the data files.
Then it will ask you to select the constants file.
3. After the script is completed, you will see two new folders in the data folder.
 - *.\output* contains plots & text files of its best guess at the stiffness, damping, and phase calculations. There is also a *new* constants file (titled *Speeds+Temps.csv*) that contains the approach speed and temperatures calculated from their associated ADC channels.
 - *.\csv* is the raw data stripped of the header and saved as csv files. Some conversions are saved as well [e.g. Temperatures are saved convert to C, Extin_csv = -Extin_raw]
4. Run *multi-excelCreate.py*. First it will ask you to select the folder that contains the csv files created in the previous step. Then it will ask you to select the constants file (use *Speeds+Temps.csv*). After the script is done, you will see a new folder within the *csv* folder, titled *xlsx-output*, that has the data & constants pasted into the manual analysis sheet. ** *Currently the location of the example manual analysis sheet is hard-coded in, you'll need to edit line 29 until I fix it.* **

I occasionally update the scripts, you can find the latest version at <https://github.com/edwardlk/Confined_Liquid_Analysis>.
