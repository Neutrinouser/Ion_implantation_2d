import sys
import csv
import os
import numpy as np
from concentration_matrix import *
from distribution import Fitting


matrixDirName = 'matrices'
figuresDirName = 'Figures' if '-figDir' not in sys.argv else sys.argv[sys.argv.index('-figDir') + 1]

# Get dataFiles
dataFiles = []
for path in sys.argv[1:]:
    if os.path.isfile(path) and path.endswith(('recorded','npy')): dataFiles.append(path)
    if os.path.isdir(path):
        for w in os.walk(os.path.join('.',path)): 
            dataFiles.extend([ os.path.join(w[0],i)  for i in w[2] if i.endswith(('recorded', 'npy')) ])

            
for dataFile in dataFiles:
    print(dataFile)
    dataFile = os.path.splitext(dataFile)[0]
    baseFileName = os.path.basename(dataFile)
    # Extract concentration matrix, transversal grid x and longitudinal grid y.
    if  matrixDirName in os.listdir() and baseFileName + '.npy' in os.listdir(matrixDirName):
        data = np.load( os.path.join(matrixDirName,baseFileName) + '.npy', allow_pickle = True).item()
        x, y, c = data['x'],data['y'], data['c']
    else:
        x,y,c = dataToConcentration(fileName=dataFile, z=7)
        if '-c' in sys.argv:   
            if matrixDirName not in os.listdir():   os.mkdir(matrixDirName) 
            np.save( os.path.join(matrixDirName,baseFileName) + '.npy', dict(x=x,y=y,c=c))
    if '-c' not in sys.argv:
        fit = Fitting(x,y,c)
        if '-simple' in sys.argv:   
            longitudinalPars, error1d = fit.simpleLongitudinalFitting() 
            if '-1d' not in sys.argv: transversalPars, error2d = fit.simpleHorizontalFitting()
        elif '-decide' in sys.argv:      
            longitudinalPars, error1d = fit.longitudinalFitting()  
            if '-1d' not in sys.argv: transversalPars, error2d = fit.horizontalFitting()
        else:
            longitudinalPars, error1d = fit.fullLongitudinalFitting()    
            if '-1d' not in sys.argv: transversalPars, error2d = fit.fullHorizontalFitting()   
        if '-1d' in sys.argv:   transversalPars = error2d = None
        fit.printing(error1d, error2d)
        if '-p' in sys.argv: 
            if figuresDirName not in os.listdir(): os.mkdir(figuresDirName)
            fit.plotting(title = baseFileName, figDir = figuresDirName,show = '-q' not in sys.argv)
    
    csvFileName = 'table.csv'
    for option in sys.argv:
        if option.endswith(('.csv', '.xlsx')):
            csvFileName = option; break
    if '-s' in sys.argv:
        csvFileName = sys.argv[sys.argv.index('-s') + 1]
        if csvFileName not in os.listdir(): open(csvFileName,'a').close()
        if os.stat(csvFileName).st_size == 0:
            with open(csvFileName,'a') as f:
                writer = csv.writer(f)
                writer.writerow(['Filename', '', 'range', 'std.dev', 'gamma', 'kurt', 'lstd.dev', 'skewxy' , 
                                    'krtxy', 'ktt', 'srange', 'sstd.dev', 'sgamm', 'skurt','lsdrp','sskewxy', 
                                    'skrtxy', 'skrtt', 'ratio', '', 'error1d', 'error2d'])
        with open(csvFileName,'a') as f:
            writer = csv.writer(f)
            pars = longitudinalPars[:4]  #+ longitudinalPars[4:8] + transversalPars[4:8] +  longitudinalPars[-1] 
            if '-1d' not in sys.argv:   pars += transversalPars[:4]
            else:   pars += ['','','','']
            if len(longitudinalPars) > 5: pars += longitudinalPars[4:8]
            else:   pars += ['','','','']
            if '-1d' not in sys.argv and len(transversalPars) > 4: pars += transversalPars[4:8]
            else:   pars += ['','','','']
            pars += [longitudinalPars[-1]]
            writer.writerow( [baseFileName] + [''] +  [str(l) for l in pars] + [''] + [str(error1d), str(error2d)])