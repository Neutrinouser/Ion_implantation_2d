import csv
import sys

csvFileName = sys.argv[-1]
convertedFile = 'converted' + csvFileName

def getExperimentalParameters(fileName):
    # File name should be of the form 
    # ./matrices/ions_Aluminium_e=800_d=1e+15_so=4_rot=0_tlt=2_recorded
    fileName = fileName.split('_')[1:-1]
    output = []
    for par in fileName:
        parVal = par.split('=')[-1]
        output.append(float(parVal) if parVal.isnumeric() else parVal)
    return output

f = open(csvFileName,'r')
g = open(convertedFile,'w')
reader, writer = csv.reader(f), csv.writer(g)
writer.writerow(['ion','energy','dose','so','rot','tlt',''] + next(reader)[2:]) 
for line in reader: 
    writer.writerow(getExperimentalParameters(line[0]) + line[1:])
f.close(), g.close()

# Sort converted csv 
reader = csv.reader(open(convertedFile,'r'))
header = next(reader)
sortedlist = list(reader)
for i in range(5,-1,-1):  sortedlist = sorted(sortedlist, key=lambda row: row[i])
with open(convertedFile,'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(sortedlist)