from lmfit import Model
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import scipy.signal
from scipy.optimize import minimize

class Fitting:
    # Introduce epsilon to be the smallest non zero concentration and define 
    # the log function so that it outputs at least log(epsilon).
    # We do this to avoid dealing with singularities across all fitting algorithms in this script.
    def __init__(self,x,y,c):
        self.x = x
        self.y = y[(y>0.05*max(y)) ]
        self.c = c[(y>0.05*max(y)) ]
        self.cVer = np.sum(self.c,axis=1)
        self.longitudinalPars = None
        self.cVerFit = None
        self.transversalPars = []
        self.cFit = np.zeros(self.c.shape)
    
    def smoothing(self,arr):
        return scipy.signal.savgol_filter(arr,31,3)

    def log(self,t):
        return np.log(np.maximum(t,np.min(self.c[self.c>0])))  

    def Pearson(self,y,mean,stdev,skew,kurt):
        # Pdf of the 1D Pearson distribution, either type IV or VI.
        # This function is almost identical to Athena's imp_prsn.C script.
        # Inputs: grid (y) and moments which are used as parameters of the distribution.
        if kurt < 0:    return self.log(np.zeros(len(y)))
        A = 10*kurt-12*skew**2-18                             
        b0 = - stdev**2*(4*kurt-3*skew**2)/A                    
        b1 = - stdev*skew*(kurt+3)/A
        b2 = - (2*kurt-3*skew**2-6)/A 
        disc = b1**2-4*b0*b2
        yC = y - mean  
        f1 = self.log(np.abs( b0 + b1 * yC + b2 * yC**2 ))/(2*b2)
        f2 = b1/b2+2*b1
        f3 = 2 * b2 * yC + b1
        f4 = np.sqrt(np.abs(disc))
        if disc>0:  #Pearson VI       
            a1,a2 = np.sort([-0.5*(b1-f4)/b2,-0.5*(b1+f4)/b2])
            f5 = np.log( np.abs((f3 - f4) / (f3 + f4)) ) 
            out = np.array(f1 - 0.5 * f2/f4 * f5) 
            out[(yC<=a1)|(yC>=a2)] = - 10**10
            #We require that a1<b1<a2
            if a1>b1 or a2<b1:
                return self.log(np.zeros(len(y)))
        elif disc<0: # Pearson IV 
            f5= np.arctan(f3/f4)
            out = np.array(f1-f2*f5/f4)   
        else: 
            out =   np.array(f1 + f2 / f3)
        # Normalise
        out -= max(out)
        out = np.exp(out); out /= np.sum(out)
        # Output is in logarithmic scale.
        return self.log(out)
    
    def dualPearson(self,y,mean,stdev,skew,kurt,meanC,stdevC,skewC,kurtC,ratio):
        if 1.15*mean> meanC:  return self.log(np.zeros(len(y)))
        out = self.log(ratio * np.exp(self.Pearson(y,mean,stdev,skew,kurt)) + (1-ratio) * np.exp(self.Pearson(y,meanC,stdevC,skewC,kurtC)))
        #plt.semilogy(y,np.exp(out)), plt.semilogy(y,self.cVer), plt.show()
        return out
    
    def simpleLongitudinalFitting(self):
        mean = np.sum(self.y*self.cVer); stdev = (np.sum(self.cVer*(self.y-mean)**2))**(0.5)
        gmodel = Model(self.Pearson)
        result = gmodel.fit( self.log(self.cVer), y=self.y , mean = mean, stdev = stdev, skew = 0, kurt = 3.5)
        self.cVerFit, self.longitudinalPars = np.exp(result.best_fit), list(result.best_values.values())
        error = np.linalg.norm(self.log(self.cVerFit) - self.log(self.cVer)) / len(self.cVer)
        return self.longitudinalPars, error

    def fullLongitudinalFitting(self):
        #Using Model from lmfit, an optimisation tool for fitting distributions, we infer 
        #optimal longitudinal parameters (moments) fitting the Pearson distribution.
        #Inputs: grid (y), cVer (vertical distribution of ion concentrations to be fitted).
        def optimize(l):
            mean = np.sum(self.y*self.cVer); 
            stdev = (np.sum(self.cVer*(self.y-mean)**2))**(0.5); skew = 0; kurt = 3.5
            mean = self.y[np.argmax(self.cVer)]
            skew = skewC = 0 
            kurt = kurtC = 3.5
            ratio = 0.99

            model = lambda y,mean,stdev,skew,kurt: self.Pearson(y=y , mean = mean, stdev = stdev, skew = skew, kurt = kurt) + np.log(np.sum(self.cVer[conds]))
            gmodel = Model(model)
            conds = (self.y < mean + l * stdev)
            result = gmodel.fit(self.log(self.cVer)[conds], y= self.y[conds] , mean = mean, stdev = stdev, skew =0 , kurt = 3.5) 
            mean,stdev,skew,kurt = list(result.best_values.values())
            # plt.semilogy(self.y[conds],np.exp(result.best_fit))

            meanC = mean + l * stdev
            stdevC = stdev
            ratio = 0.99

            model = lambda y,mean,stdev,meanC,stdevC, ratio, skew, kurt: self.dualPearson(y=y , mean = mean, stdev = stdev, skew = skew, kurt = kurt,\
                                                    meanC = meanC, stdevC = stdevC, skewC = 0,kurtC = 3.5,ratio = ratio) + (1-ratio)**2
            gmodel = Model(model)
            result1 = gmodel.fit( self.log(self.cVer), y= self.y , mean = mean, stdev = stdev, meanC = 1.25*mean,stdevC = stdev,ratio=ratio,skew = skew,kurt = kurt) 
            mean,stdev,meanC,stdevC, ratio, skew, kurt = list(result1.best_values.values())
            
            model = lambda y,mean,stdev,meanC,stdevC, ratio: self.dualPearson(y=y , mean = mean, stdev = stdev, skew = skew, kurt = kurt,\
                                        meanC = meanC, stdevC = stdevC, skewC = skewC,kurtC = kurtC,ratio = ratio)
            gmodel = Model(model)
            result1 = gmodel.fit( self.log(self.cVer), y= self.y , mean = mean, stdev = stdev, meanC = meanC,stdevC = stdevC,ratio=ratio) 
            mean,stdev,meanC,stdevC, ratio = list(result1.best_values.values())
            
            # modelFun=self.dualPearson
            gmodel = Model(self.dualPearson)
            gmodel.set_param_hint('ratio', min=0.8, max = 1)
            # gmodel.set_param_hint('mean', min=0.8, max = 1)
            result = gmodel.fit( self.log(self.cVer), y= self.y , mean = mean, stdev = stdev, skew = skew, kurt = kurt,\
                                                    meanC = meanC, stdevC = stdevC, skewC = skewC,kurtC = kurtC,ratio = ratio)
            return  np.exp(result.best_fit), list(result.best_values.values())
        def logRSS(l):  
            cVerFit = optimize(l)[0]
            return np.linalg.norm(self.log(cVerFit) - self.log(self.cVer)) / len(self.cVer)
        errors = []
        for l in np.linspace(0,2,4):
            errors.append((logRSS(l),l))
        l = min(errors)[1]
            
        # l = minimize(logRSS,(2,), bounds = ((0,3),)).x[0]
        self.cVerFit, self.longitudinalPars = optimize(l)
        return self.longitudinalPars, min(errors)[0]

    def longitudinalFitting(self):
        _ , errorSimple = self.simpleLongitudinalFitting()
        _ , errorFull = self.fullLongitudinalFitting()
        self.mode = 'Full'
        if errorFull > 2/3 * errorSimple:
            _ , error = self.simpleLongitudinalFitting()
            self.mode = 'Simple'
        else:   error = errorFull
        return self.longitudinalPars,error

    # For transversal fitting, you may consult 
    # 1. Two-dimensional modelling of ion implantation with spatial moments, pages 453-454.
    # 2. Athena's implemantation code imp_lat.C
    def MGF(self,x,mean,stdev,beta):
        # Pdf of the Modified Gaussian distribution as defined in the paper above.
        invp0= 0.290576* np.sqrt(beta-1.8); invpInf= 0.687042* np.log(np.sqrt(5)*beta/3)
        w= 0.795833 * np.exp(-1.94544*(beta-1.8)) + 0.204167 * np.exp(-0.272172*(beta-1.8))
        p = 1/(w*invp0 + (1-w)*invpInf)
        logdist = -np.abs(1/stdev*np.sqrt(gamma(3/p)/gamma(1/p)) * (x-mean))**p 
        logdist -= max(logdist)
        return np.exp(logdist)
    def PearsonII(self,x,mean,stdev,beta):
        # Pdf of the Pearson type II distribution.
        logdist = np.zeros(len(x))
        domain = (mean - np.sqrt(2*beta/(3-beta))*stdev < x) & (x <  mean + np.sqrt(2*beta/(3-beta))*stdev)
        logdist[domain] = np.log(np.abs(1+(beta-3)/(2*beta*stdev**2)*(x[domain]-mean)**2))*(0.5*(5*beta-9)/(3-beta))
        if domain.any():    logdist[domain] -= max(logdist[domain])
        return np.exp(logdist) * domain

    def horizontalDistribution(self,x,mean,stdev,kurt):
        # Outputs the transversal pdf in the grid given by x corresponding to the transversal moment parameters. 
        #If 3<kurt<6 pdf is MGF, else if 1.8<kurt<3 average of MGF and Pearson II
        if stdev>0:
            #Trans kurt needs to be greater than 1.8 and stdev needs to be positive
            if (kurt<3) and (kurt>1.8):
                mgf = self.MGF(x,mean,stdev,kurt)
                pII = self.PearsonII(x,mean,stdev,kurt) 
                if np.sum(mgf)!=0 and np.sum(pII)!=0:
                    return 0.5 * mgf/np.sum(mgf) + 0.5 * pII/np.sum(pII)
            if kurt>=3:
                mgf = self.MGF(x,mean,stdev,kurt)
                if np.sum(mgf)!=0:
                    return  mgf/np.sum(mgf)
        return np.zeros(len(x))

    def simpleHorizontalFitting(self,cVerFit = None, longitudinalPars = None, secondaryFitting = False):
        # Performs parameter optimisation and outputs optimal transversal moments.
        # Inputs: the (x,y) spatial grid and the concentration matrix to be fitted, the vertical fit which we will 
        # spread horizontally by multiplying it with the transversal (depth-dependent) distribution, and longitudinal
        # parameters used in converting the quadratic coefficients to 3 of the 4 trans moments.
        # The last one, transversal kurtosis is assumed to be depth indendent.
        # For more information, look at Athena manual pages 3-73 to 3-75; 
        # in particulat about the depth-dependent (quadratic) trans stdev.

        if cVerFit is None: cVerFit = self.cVerFit
        if longitudinalPars is None:    longitudinalPars = self.longitudinalPars[:4]
        # Extract vertical moments
        meanVer, stdevVer, skewVer, kurtVer = longitudinalPars
        # Define the model of the two dimensional pdf used to fit the concentration matrix c.
        def twoDPdf(xyGrid, kurt,c0,c1,c2): 
            # print(kurt,c0,c1,c2)
            # xyGrid is only needed for the optimisation to run properly.  
            # Quadratically dependent transversal stdev 
            stdevXArray = np.sqrt(np.maximum( c0 + c1 * (self.y - meanVer) + c2 * (self.y - meanVer)**2 ,0)) 
            # Centering (supposedly symmetric) distribution; very helpful for Monte Carlo simulations with tilt.
            meanX = [np.sum(cHor*self.x)/np.sum(cHor) if np.sum(cHor)!=0 else 0  for cHor in self.c] 
            # twoD fitted distribution = vertical fit X (depth dependent) horizontal fit.
            cFit = np.array([cV*self.horizontalDistribution(self.x,m,s,kurt) for m,s,cV in zip(meanX,stdevXArray,cVerFit)])
            # if c1 == 0 and c2 == 0:    X, Y = np.meshgrid(self.x, self.y); plt.contourf(X,Y,self.log(cFit)), plt.show()
            return self.log(cFit)
        # Optimisation in logarithmic scale.
        X, Y = np.meshgrid(self.x, self.y)
        if secondaryFitting:
            twoDPdfS = lambda xyGrid, kurt,c0: twoDPdf(xyGrid, kurt,c0,c1=0,c2=0)
            gmodel = Model(twoDPdfS)
            gmodel.set_param_hint('c0', min=0)
            gmodel.set_param_hint('kurt', min=1.8, max = 6.0)
            result = gmodel.fit( self.log(self.c - self.cFit) ,xyGrid=X+Y*1j ,kurt= 2.9  ,c0= (np.max(self.x)/3)**2)
            kurtHor,c0 = result.best_values.values()
            c1 = c2 = 0 
        else: #primary horizontal fitting
            gmodel = Model(twoDPdf)
            # gmodel.set_param_hint('c0', min=0)
            gmodel.set_param_hint('kurt', min=1.8, max = 6.0)
            result = gmodel.fit( self.log(self.c - self.cFit) ,xyGrid=X+Y*1j ,kurt= 2.9  ,c0= (np.max(self.x)/3)**2,c1=0,c2=0)
            kurtHor,c0,c1,c2 = result.best_values.values(); 
        coefs = [c0,c1 * stdevVer,c2 * stdevVer**2]
        # Convert quadratic coefficients to mixed moments and average trans stdev.
        stdevHor = np.sqrt(np.dot([1,0,1],coefs))
        skewXY = np.dot([0,1,skewVer],coefs)/stdevHor**2
        kurtXY = np.dot([1,skewVer,kurtVer],coefs)/stdevHor**2
        # Ouput parameters and the 2d fit.

        # plt.subplot(2,1,1), plt.contourf(X,Y,self.log(self.c - self.cFit))
        # plt.subplot(2,1,2), plt.contourf(X,Y,result.best_fit )
        # plt.show()

        self.transversalPars.extend([stdevHor,skewXY,kurtXY,kurtHor])
        self.cFit += np.exp(result.best_fit) 
        return self.transversalPars, np.sqrt( np.sum((self.log(self.cFit) - self.log(self.c))**2 ))  / np.prod(self.c.shape) 

    def fullHorizontalFitting(self):
        # First decompose cVerFit = cVerFitPrimary + sVerFitSecondary
        meanP,stdevP,skewP,kurtP,meanS,stdevS,skewS,kurtS, ratio = self.longitudinalPars
        cVerFitPrimary = np.exp(self.Pearson(self.y,meanP,stdevP,skewP,kurtP)) * ratio
        cVerFitSecondary = np.exp(self.Pearson(self.y,meanS,stdevS,skewS,kurtS)) * (1-ratio)

        # Fit the primary horizontal distribution first
        self.simpleHorizontalFitting(cVerFit = cVerFitPrimary)

        # Fit the secondary horizonal distribution   
        self.simpleHorizontalFitting(cVerFit = cVerFitSecondary, longitudinalPars = self.longitudinalPars[4:8], secondaryFitting = True )

        return self.transversalPars, np.sqrt( np.sum((self.log(self.cFit) - self.log(self.c))**2 ))  / np.prod(self.c.shape) 
    
    def horizontalFitting(self):
        if self.mode == 'Simple':    return self.simpleHorizontalFitting()
        if self.mode == 'Full':  return self.fullHorizontalFitting()

    def printing(self,error1d, error2d):
        if self.cVerFit is not None:    
            print('Longitudinal parameters: ' + str(self.longitudinalPars))
            print('Longitudinal error percentage: ' + str(100 * error1d) + '%.')
        if self.transversalPars:
            print('Transversal parameters: ' + str(self.transversalPars))
            # The error is defined to be the Frobenius norm of the logarithmic residual matrix divided by the size of the matrix.
            print('2d error percentage: ' + str(100 * error2d) + '%.')
        

    def plotting(self, title='No filename given',figDir = 'Figures', show = True):
        if self.cVerFit is not None:
            plt.semilogy(self.y,self.cVer), plt.semilogy(self.y,self.cVerFit)
            plt.xlabel(r'$y$'), plt.title(r'Longitudinal distribution')
            plt.title(title)
            plt.savefig(figDir + '/' + title + '_1d' + '.png')
            if show:    plt.show()
            plt.close()
        # For 2D plotting
        if self.transversalPars:
            X, Y = np.meshgrid(self.x, self.y)
            plt.subplot(3,1,1),plt.contourf(X,Y,self.log(self.cFit))
            plt.subplot(3,1,2),plt.contourf(X,Y,self.log(self.c))
            plt.subplot(3,1,3),plt.contourf(X,Y,self.log(self.cFit) - self.log(self.c))
            plt.savefig(figDir + '/' + title + '_2d' + '.png')
            if show:    plt.show()
            plt.close()