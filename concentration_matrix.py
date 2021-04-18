import numpy as np
from scipy.stats import zscore

def dataToConcentration(fileName,z=7):
    #Loads data from the selected file
    with open(fileName,'r') as h:
        data = [list(map(float,line.split(' ')[:-1])) for line in h.readlines() ]
    # Define horizontal, vertical displacements and weights 
    # (weights express the number of ions corresponding to each observation)

    # There are two horizontal axis; we control the transversal direction by the angle
    angle = np.pi/2
    horDisp  = np.array([np.cos(angle)*(row[4] - row[1]) + np.sin(angle)*(row[5]-row[2]) for row in data])
    verDisp  = np.array([row[0] - row[3] for row in data])
    weights = np.array([row[6] for row in data])

    #Getting rid of the outliers and ions with negative vertical displacement (out of the wafer)
    def applyFilters(hor,ver,weights,maxZScore):
        # Longitudinal direction: get rid of all outliers at high depths 
        # (keep the ones near 0 even if they are outliers)
        maxY = np.max(ver[np.abs(zscore(ver)) < maxZScore]); l1 = ver <= maxY
        # Get rid of transversal direction outliers
        l2 = np.abs(zscore(hor)) < maxZScore
        # Get rid of reflected ions
        l3 = np.array(ver)>=0
        # Output
        nonOutliers = l1 & l2 & l3 
        return hor[nonOutliers], ver[nonOutliers], weights[nonOutliers]

    horDisp, verDisp, weights = applyFilters(horDisp, verDisp, weights, maxZScore = z)

    #Get concentration matrix adjusted by weight; use histogram 2d.
    horRange = [min(horDisp),max(horDisp)]
    verRange = [0,max(verDisp)]
    binsDims = [100,200]
    weightsSum = np.sum(weights)
    concentrationMat = np.zeros(binsDims) 
    for weight in set(weights):
        numberOfIons = list(weights).count(weight)
        hor = horDisp[weights==weight]   
        ver = verDisp[weights==weight]   
        hist, xedges, yedges = np.histogram2d(hor,ver,bins=binsDims,range=[horRange, verRange])
        concentrationMat += numberOfIons * weight/weightsSum * hist/np.sum(hist)

    return xedges[:-1], yedges[:-1], np.transpose(concentrationMat)
