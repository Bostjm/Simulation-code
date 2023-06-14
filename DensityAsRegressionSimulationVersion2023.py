from numpy import *
import pandas as pd
import time
import os
import scipy.special as sc
from scipy import integrate
from scipy import interpolate
from scipy.stats import norm
import tensorflow as tf
import tensorflow_model_optimization.sparsity as tfmotspar
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

####################
#REMARKS AT THE START
####################
#For the article, the code has been run using:
#Python 3.9.5
#Numpy version 1.20.3
#Pandas version 1.2.4
#SciPy version 1.6.3
#scikit-learn version 1.2.0
#TensorFlow version 2.6.0 including Keras 2.6.0, built with cuDNN 8.2.1.32 and CUDA 11.3.1
#TensorFlow-model-optimization 0.7.3

####################
#BROWNIAN MOTION RELATED FUNCTIONS
####################

### Geometric Brownian Motion generation
def GBMPathGeneration(Steps, initial, Seed):
    """
    Function to create an approximate Geometric Brownian Motion path
    """
    rng=random.default_rng(Seed) #Create the (seeded) random generator
    Stepsize = 1/Steps #Calculate the stepsize
    #Calculate the Brownian motion path
    y = norm.rvs(scale= sqrt(Stepsize), size = Steps, random_state = rng) #Draw the required number of normal random variables
    y = insert(arr = y, obj= 0 , values =initial ) #Add the initial value
    z = cumsum(y)
    gbm = exp(z)
    return(gbm)

### Calculate the values in the step-points of the function x(1-x) exp(W(x)) where W is the Brownian motion (so exp(W) is the Geometric BM)
def GBMNormalizedBoundaryVariationDensity(GBMPath, Steps):
    """
    Function to multiply the Geometric Brownian Motion path with a (scaled) version of x(1-x) so it satisfies the boundary conditions. Furthermore it creates a linear interpolation and normalizes the resulting function so it becomes a density
    """
    GBMbv= zeros(Steps+1)
    for i in range (0,Steps+1):
        GBMbv[i] = GBMPath[i]*(i/Steps)*(1-i/Steps)
    t = linspace(0.0,1, Steps+1)
    I = sum(diff(t)*(GBMbv[:-1]+diff(GBMbv)/2)) #We use the fact that after linear interpolation our function is piecewise linear, so we can directly compute the value of the integral on each piece)
    print(I)
    z = GBMbv/I
    f = interpolate.interp1d(t, z)
    w = f(t)
    print(sum(diff(t)*(w[:-1]+diff(w)/2)) )
    return f

def GBMNormalizedConcentratedVariationDensity(GBMPath, Steps):
    """
    Variation of GBMNormalizedBoundaryVariationDensity that concentrates the function on 3/4-th of the interval. Basis for the shifting conditional density.
    """
    Stepsize = 3/(4*Steps) #Calculate the stepsize, the factor 3/4 is because we want to concentrate the function on [0,3/4]
    GBMncv= zeros(Steps+1)
    for i in range (0,Steps+1):
        GBMncv[i] = GBMPath[i]*((4/3)*i*Stepsize)*(1-(4/3)*i*Stepsize)
    t = linspace(0.0, 3/4, Steps+1)
    I = sum(diff(t)*(GBMncv[:-1]+diff(GBMncv)/2))
    t = append(t,1) #Add the point 1 at the end to extend interpolation to [0,1]
    z = GBMncv/I
    z = append(z,0) #Add the functionvalue 0 to the point 1
    f = interpolate.interp1d(t,z) #Create an interpolation function
    w = f(t)
    print(sum(diff(t)*(w[:-1]+diff(w)/2)) )
    return f


def GBMDensity(GBMPath, Steps):
    """
    Create a linear interpolation of the approximate GBM and normalize it to make it a density. Before this function is only defined in 2^k+1 points, after this function is a density defined for points on the entire interval.
    """
    t = linspace(0.0,1, Steps+1)
    I = sum(diff(t)*(GBMPath[:-1]+diff(GBMPath)/2))
    z = GBMPath/I
    f = interpolate.interp1d(t, z)
    w = f(t)
    print(sum(diff(t)*(w[:-1]+diff(w)/2)) )
    return f


def GBMNormalizedMarginal(NumberOfPoints, Seed1, Seed2, f, Bound):
    """
    Rejection sampling function that samples from an 1-d density function f.
    """
    rng=random.default_rng(Seed1) #Create the (seeded) random generator
    rng2=random.default_rng(Seed2) #Create a second (seeded) random generator
    i=0
    X = pd.DataFrame(columns=['X'], dtype='float64')
    while i < NumberOfPoints:
        T = rng.random(1) #Draw sample from the uniform distribution on [0,1)
        
        A = rng2.random(1) #Draw sample from the uniform distribution on [0,1)
        if Bound*A<=f(T):
            AidFrame = pd.DataFrame({"X": T})
            X = pd.concat([X,AidFrame], ignore_index = True, axis=0 )
            i = i+1
    return X


####################
#MARGINAL DENSITY FUNCTIONS
####################

#Density with smoothness 1/2 to be used with the copula model

def CopulaMarginalDensity(x,dimension):
    """
    Univariate density used in the copula model. This density is between 1-1/(2d) and 1+1/(2d).
    """
    if x<1/4:
        y = 1+(1/(2*dimension))-(1/dimension)*sqrt(1/4-x)
    elif x < 1/2:
        y = 1+(1/(2*dimension))-(1/dimension)*sqrt(x-1/4)
    elif x < 3/4:
        y = 1-(1/(2*dimension))+(1/dimension)*sqrt(3/4-x)
    else:
        y = 1-(1/(2*dimension))+(1/dimension)*sqrt(x-3/4)
    return y

def CopulaMarginalCumulative(x, dimension):
    """
    The cumulative distribution function belonging to the univariate density for the copula model.
    """
    if x<1/4:
        y = x*(1+(1/(2*dimension)))+(2/(3*dimension))*((1/4-x)**(3/2))-1/(12*dimension)
    elif x < 1/2:
        y = x*(1+(1/(2*dimension)))-(2/(3*dimension))*((x-1/4)**(3/2))-1/(12*dimension)
    elif x < 3/4:
        y = x*(1-(1/(2*dimension)))-(2/(3*dimension))*((3/4-x)**(3/2))+5/(12*dimension)
    else:
        y = x*(1-(1/(2*dimension)))+(2/(3*dimension))*((x-3/4)**(3/2))+5/(12*dimension)
    return y


def NearUnifDensity(dimension):
    """
    Create a simple function that is close to a uniform distribution. This function is stored in the same form as the GBM based functions.
    These functions are used to prevent the L-infinity norm from exploding in higher dimension (as is the case when only GBM based functions are used).
    The product of these functions has nice behaviour when the dimension goes to infinity: (1-1/d)^d goes to 1/e while (1+1/d)^d goes to e.
    """
    t = (0.0,1.0)
    x = 1/dimension #Notice that the dimension is always >=2 when this function is used, thus function is always between 0.5 and 1.5.
    y = (1-x,1+x)
    f = interpolate.interp1d(t,y) #Create an interpolation function
    return f

####################
#MARGINAL SAMPLING FUNCTIONS
####################

def Uniform(NumberOfPoints, Seed):
    rng=random.default_rng(Seed) #Create the (seeded) random generator
    X= rng.random(NumberOfPoints)   #Draw NumberOfPoints Samples from the uniform distribution on [0,1)
    MarginalData = pd.DataFrame({"X": X[:]})
    return MarginalData

####################
#CONDITIONAL DENSITY FUNCTIONS
####################

def GBMMixingConditionalDensity(f,x, ConditionalValue):
    """
    Conditional density where the value of the conditional variable determines the weight assigned to f(x) or its mirrored version f(1-x)
    """
    return (1.0-ConditionalValue)*f(x)+(ConditionalValue)*f(1-x) 

def GBMShiftingConditionalDensity(f, x, ConditionalValue):
    """
    Condition density where the conditional variabel determines the location of the function f on the interval. Only to be used in combination with the concentrated densities f
    """
    y = max(x-ConditionalValue/4,0)
    return f(y)

####################
#CONDITIONAL SAMPLING FUNCTIONS
####################


def GBMMixingConditional(NumberOfPoints, ConditionalValues, Seed1, Seed2, f, Bound): 
    """
    Function to do rejection sampling from the mixing conditional density
    """
    rng=random.default_rng(Seed1) #Create the (seeded) random generator
    rng2=random.default_rng(Seed2) #Create a second (seeded) random generator
    i=0
    X = pd.DataFrame(columns=['X'], dtype='float64')
    while i < NumberOfPoints:
        T = rng.random(1) #Draw sample from the uniform distribution on [0,1)
        A = rng2.random(1) #Draw sample from the uniform distribution on [0,1)
        Aid = GBMMixingConditionalDensity(f,T, ConditionalValues.iloc[i,0], )
        if Bound*A<= Aid:
            AidFrame = pd.DataFrame({"X": T})
            X = pd.concat([X,AidFrame], ignore_index = True, axis=0 )
            i = i+1
    return X

def GBMShiftingConditional(NumberOfPoints, ConditionalValues, Seed1, Seed2, f, Bound): 
    """
    Function to do rejection sampling from the shifting conditional density
    """
    rng=random.default_rng(Seed1) #Create the (seeded) random generator
    rng2=random.default_rng(Seed2) #Create a second (seeded) random generator
    i=0
    X = pd.DataFrame(columns=['X'], dtype='float64')
    while i < NumberOfPoints:
        T = rng.random(1) #Draw sample from the uniform distribution on [0,1)
        A = rng2.random(1) #Draw sample from the uniform distribution on [0,1)
        Aid = GBMShiftingConditionalDensity(f,T,ConditionalValues.iloc[i,0])
        if Bound*A<= Aid:
            AidFrame = pd.DataFrame({"X": T})
            X = pd.concat([X,AidFrame], ignore_index = True, axis=0 )
            i = i+1
    return X

####################
#MODELS
####################


#####
#'NaiveBayesGBM'
#####

def NaiveBayesGBMDensity(NumberOfPoints, Dimension ,ChildSeeds, Test: bool, SimulationNumber, f, Init: bool, Bound, ConditionalModel):
    """
    Function to create data for the Naive Bayes model.
    """
    if Init:
        index = int(2400+SimulationNumber*100) #Goal of index is to ensure that we generate a different data sample for each simulationnumber, while still being reproducible
    elif Test:
        index = int(1900+SimulationNumber*100)
    else:
        index = int(1950+SimulationNumber*100)
    X =  GBMNormalizedMarginal(NumberOfPoints, ChildSeeds[index], ChildSeeds[index+1], f[0], Bound[0])
    if ConditionalModel == 'Mixing':
        if Dimension >= 2:
            Y =  GBMMixingConditional(NumberOfPoints, X, ChildSeeds[index+2], ChildSeeds[index+3], f[1], Bound[1])
            Z = pd.concat([X,Y], axis=1, join="inner")
        else:
            Z = X
        for d in range(2,Dimension): #Start at 2 since we already did the first two
            Y = GBMMixingConditional(NumberOfPoints, X, ChildSeeds[index+2*d], ChildSeeds[index+1+2*d], f[d], Bound[d])
            Z = pd.concat([Z,Y], axis=1, join="inner")
    elif ConditionalModel == 'Shifting':
        if Dimension >= 2:
            Y =  GBMMixingConditional(NumberOfPoints, X, ChildSeeds[index+2], ChildSeeds[index+3], f[1], Bound[1]) #The second dimension is always a near uniform density for which shifting does not make sense
            Z = pd.concat([X,Y], axis=1, join="inner")
        else:
            Z = X
        for d in range(2,Dimension): #Start at 2 since we already did the first two
            if d%3 > 0: #Only use the shifting conditional if the marginal is based on the concentrated GBM
                Y = GBMMixingConditional(NumberOfPoints, X, ChildSeeds[index+2*d], ChildSeeds[index+1+2*d], f[d], Bound[d])
            else:
                Y = GBMShiftingConditional(NumberOfPoints, X, ChildSeeds[index+2*d], ChildSeeds[index+1+2*d], f[d], Bound[d])
            Z = pd.concat([Z,Y], axis=1, join="inner")   
    return Z

def NaiveBayesGBMTrueValues(Datapoints: pd.DataFrame, NumberOfPoints, Dimension, f, ConditionalModel):
    """
    Function that provides the true values of the density for the testdata for the Naive Bayes Model
    """
    Z = zeros(NumberOfPoints)
    for i in range(0, NumberOfPoints):
        Aid = f[0](Datapoints.iloc[i,0])
        if ConditionalModel == 'Mixing':
            for j in range(1, Dimension):
                Aid = Aid*GBMMixingConditionalDensity(f[j],Datapoints.iloc[i,j],Datapoints.iloc[i,0])
            Z[i] = Aid
        elif ConditionalModel == 'Shifting':
            for j in range(1, Dimension):
                if j%3 > 0: #Only use the shifting conditional if the marginal is based on the concentrated GBM
                    Aid = Aid*GBMMixingConditionalDensity(f[j],Datapoints.iloc[i,j],Datapoints.iloc[i,0])
                else:
                    Aid = Aid*GBMShiftingConditionalDensity(f[j],Datapoints.iloc[i,j],Datapoints.iloc[i,0])
            Z[i] = Aid
    return pd.DataFrame({"Z": Z[:]})


#####
#'BayesBinaryTree'
#####
#Bayesian network in binary tree form

def BayesBinaryTreeDensity(NumberOfPoints, Dimension ,ChildSeeds, Test: bool, SimulationNumber, f, Init: bool, Bound, ConditionalModel):
    """
    Function to create data for the tree-shaped Bayesian network model.
    """
    if Init:
        index = int(3400+SimulationNumber*100)
    elif Test:
        index = int(2900+SimulationNumber*100)
    else:
        index = int(2950+SimulationNumber*100) 
    X =  GBMNormalizedMarginal(NumberOfPoints, ChildSeeds[index], ChildSeeds[index+1], f[0], Bound[0])
    if ConditionalModel == 'Mixing':
        if Dimension >= 2:
            Y =  GBMMixingConditional(NumberOfPoints, X, ChildSeeds[index+2], ChildSeeds[index+3], f[1], Bound[1])
            Z = pd.concat([X,Y], axis=1, join="inner")
        else:
            Z = X
        for d in range(2,Dimension):
            Cond = Z.iloc[:,int(floor((d-1)/2))]
            Cond = Cond.to_frame()
            Y = GBMMixingConditional(NumberOfPoints, Cond, ChildSeeds[index+2*d], ChildSeeds[index+1+2*d],  f[d], Bound[d]) 
            Z = pd.concat([Z,Y], axis=1, join="inner")
    elif ConditionalModel == 'Shifting':
        if Dimension >= 2:
            Y =  GBMMixingConditional(NumberOfPoints, X, ChildSeeds[index+2], ChildSeeds[index+3], f[1], Bound[1])#The second dimension is always a near uniform density for which shifting does not make sense
            Z = pd.concat([X,Y], axis=1, join="inner")
        else:
            Z = X
        for d in range(2,Dimension):
            Cond = Z.iloc[:,int(floor((d-1)/2))]
            Cond = Cond.to_frame()
            if d%3 > 0: #Only use the shifting conditional if the marginal is based on the concentrated GBM
                Y = GBMMixingConditional(NumberOfPoints, Cond, ChildSeeds[index+2*d], ChildSeeds[index+1+2*d],  f[d], Bound[d])
            else:
                Y = GBMShiftingConditional(NumberOfPoints, Cond, ChildSeeds[index+2*d], ChildSeeds[index+1+2*d],  f[d], Bound[d])
            Z = pd.concat([Z,Y], axis=1, join="inner")
    return Z

def BayesBinaryTreeTrueValues(Datapoints: pd.DataFrame, NumberOfPoints, Dimension, f,ConditionalModel):
    """
    Function that provides the true values of the density for the testdata for the tree-shaped Bayesian network model
    """
    Z = zeros(NumberOfPoints)
    for i in range(0, NumberOfPoints):
        Aid = f[0](Datapoints.iloc[i,0])
        if ConditionalModel == 'Mixing':
            for j in range(1, Dimension):
                Aid = Aid*GBMMixingConditionalDensity(f[j],Datapoints.iloc[i,j],Datapoints.iloc[i,int(floor((j-1)/2))]) 
            Z[i] = Aid
        elif ConditionalModel == 'Shifting':
            for j in range(1, Dimension):
                if j%3 > 0: #Only use the shifting conditional if the marginal is based on the concentrated GBM
                    Aid = Aid*GBMMixingConditionalDensity(f[j],Datapoints.iloc[i,j],Datapoints.iloc[i,int(floor((j-1)/2))])
                else:
                    Aid = Aid*GBMShiftingConditionalDensity(f[j],Datapoints.iloc[i,j],Datapoints.iloc[i,int(floor((j-1)/2))])
            Z[i] = Aid
    return pd.DataFrame({"Z": Z[:]})

#####
#'Copula'
#####
#Vine Copula Based model using the Farlie-Gumbel-Morgnestern Copula (family) 

def CopulaDensityFunction(X, Dimension, parameter):
    """
    The joint density function for the copula model.
    """
    Y = zeros(Dimension)
    CDFY = zeros(Dimension)
    Copula = ones(Dimension-1)
    for d in range (0,Dimension):
        Y[d] = CopulaMarginalDensity(X[d], Dimension)
        CDFY[d] = CopulaMarginalCumulative(X[d], Dimension)
        if d >= 1:
            Copula[d-1] = 1+parameter[d-1]*(1-2*CDFY[d-1])*(1-2*CDFY[d])
    Joint = prod(Y)*prod(Copula)
    return  Joint

def CopulaDensity(NumberOfPoints, Dimension ,ChildSeeds, Test: bool, SimulationNumber, Init: bool, CopulaParameter):
    """
    Function to create data for the copula model
    """
    if Init:
        index = int(1400+SimulationNumber*100)
    elif Test:
        index = int(900+SimulationNumber*100)
    else:
        index = int(950+SimulationNumber*100)
    rng=random.default_rng(ChildSeeds[index]) #Create the (seeded) random generator
    rng2=random.default_rng(ChildSeeds[index+1]) #Create a second (seeded) random generator
    i=0
    Bound = 1+1/(2*Dimension) #The Copula marginal density is bounded by 1+1/(2d)
    for d in range(1,Dimension): #Start at 1 first we already have done the first
        Bound = Bound*(1+1/(2*Dimension))*(1+abs(CopulaParameter[d-1])) #Each dimension we multiple with a marginal bounded by 1.8 and a Copulafunction bounded by 1 + absolute value of the copulaparameter
    X = pd.DataFrame(dtype='float64')
    while i < NumberOfPoints:
        T = rng.random(Dimension) #Draw sample from the uniform distribution on [0,1)^d
        A = rng2.random(1) #Draw sample from the uniform distribution on [0,1)
        if Bound*A<=CopulaDensityFunction(T,Dimension,CopulaParameter):
            AidFrame = pd.DataFrame([T])
            X = pd.concat([X,AidFrame], ignore_index = True, axis=0 )
            i = i+1
    return X


def CopulaTrueValues(Datapoints: pd.DataFrame, NumberOfPoints, Dimension, CopulaParameter):
    """
    Function that provides the true values of the density for the testdata for the tree-shaped Bayesian network model
    """
    Z = zeros(NumberOfPoints)
    for i in range(0, NumberOfPoints):
        Datapoint = Datapoints.iloc[i,:]
        Datapoint = Datapoint.to_numpy()
        Z[i] = CopulaDensityFunction(Datapoint, Dimension, CopulaParameter)
    return pd.DataFrame({"Z": Z[:]})


####################
#DATA GENERATION
####################
#Place to generate the data and split it in regression and kernel sets.

def GenerateRegressionData(DataPoints: pd.DataFrame, NumberOfPoints, Dimension, BandwidthConstantRegression):
    """
    Function that splits up the data, uses one half to create a kernel estimator and the other to generate regression data from that estimator. This is the approach used for the theory.
    """
    splitvalue = int(NumberOfPoints/2)
    X = DataPoints.iloc[:splitvalue,:]
    Y = DataPoints.iloc[splitvalue:,:]
    Y = Y.reset_index(drop=True)
    Z, Bandwidth = KernelEstimationRegression(X,Y, splitvalue, Dimension, BandwidthConstantRegression)
    return pd.concat([X,Z], axis=1), Bandwidth

def GenerateRegressionDataFullDataMethod(DataPoints: pd.DataFrame, NumberOfPoints, Dimension, BandwidthConstantRegression):
    """
    Function that creates a kernel estimator using all the data and generates regression data from that estimator using the same data. This is the full data method.
    """
    X = DataPoints
    Y = DataPoints
    Y = Y.reset_index(drop=True)
    Z, Bandwidth = KernelEstimationRegression(X,Y, NumberOfPoints, Dimension, BandwidthConstantRegression)
    return pd.concat([X,Z], axis=1), Bandwidth

def GenerateTestDataGBM(NumberOfPoints, Dimension, ChildSeeds, SimulationNumber,f, Bound,ConditionalModel):
    """
    Function to generate test data for the Naive Bayes model
    """
    DataPoints = NaiveBayesGBMDensity(NumberOfPoints, Dimension, ChildSeeds, True, SimulationNumber, f, False, Bound,ConditionalModel)
    TrueValues = NaiveBayesGBMTrueValues(DataPoints, NumberOfPoints, Dimension, f,ConditionalModel)
    return pd.concat([DataPoints,TrueValues], axis=1, join="inner")

def GenerateTestDataBinaryTree(NumberOfPoints, Dimension, ChildSeeds, SimulationNumber,f, Bound,ConditionalModel):
    """
    Function to generate test data for the tree-shaped Bayesian network model
    """
    DataPoints = BayesBinaryTreeDensity(NumberOfPoints, Dimension, ChildSeeds, True, SimulationNumber, f, False, Bound,ConditionalModel)
    TrueValues = BayesBinaryTreeTrueValues(DataPoints, NumberOfPoints, Dimension, f,ConditionalModel)
    return pd.concat([DataPoints,TrueValues], axis=1, join="inner")

def GenerateTestDataCopula(NumberOfPoints, Dimension, ChildSeeds, SimulationNumber,CopulaParameter):
    """
    Function to generate test data for the copula model
    """
    DataPoints = CopulaDensity(NumberOfPoints, Dimension, ChildSeeds, True, SimulationNumber, False,  CopulaParameter)
    TrueValues = CopulaTrueValues(DataPoints, NumberOfPoints, Dimension, CopulaParameter)
    return pd.concat([DataPoints,TrueValues], axis=1, join="inner")

#Main function for the data generation 
def GenerateData(Dimension, NumberOfPoints, SimulationNumber, ChildSeeds, TestDataNumber, f, Model, BandwidthConstantRegression, BoundGBM,ConditionalModel,CopulaParameter):
    """
    Function that checks which model is used and generates the data, the regression data and the testdata for that model.
    """
    if Model == 'NaiveBayesGBM':
        FullData = NaiveBayesGBMDensity(NumberOfPoints,Dimension,ChildSeeds, False, SimulationNumber, f, False, BoundGBM,ConditionalModel)
    elif Model == 'BayesBinaryTree':
        FullData = BayesBinaryTreeDensity(NumberOfPoints,Dimension,ChildSeeds, False, SimulationNumber, f, False, BoundGBM,ConditionalModel)
    elif Model == 'Copula':
        print("Generating data copula")
        FullData = CopulaDensity(NumberOfPoints,Dimension,ChildSeeds, False, SimulationNumber, False,CopulaParameter)  
    RegressionKDERunTime = 0
    start = time.time()
    print("Generating regression data")
    RegressionData, RegressionKDEBandwidth = GenerateRegressionData(FullData, NumberOfPoints, Dimension, BandwidthConstantRegression)
    end = time.time()
    RegressionKDERunTime = float(end-start)
    RegressionDataFullDataMethod, RegressionKDEFullBandwidth = GenerateRegressionDataFullDataMethod(FullData, NumberOfPoints, Dimension, BandwidthConstantRegression)
    if Model == 'NaiveBayesGBM':
        TestData = GenerateTestDataGBM(TestDataNumber, Dimension, ChildSeeds, SimulationNumber,f, BoundGBM,ConditionalModel)  
    elif Model == 'BayesBinaryTree' :
        TestData = GenerateTestDataBinaryTree(TestDataNumber, Dimension, ChildSeeds, SimulationNumber,f, BoundGBM,ConditionalModel) 
    elif Model == 'Copula' :
        print("Generating testdata copula")
        starttestdatagen = time.time()
        TestData = GenerateTestDataCopula(TestDataNumber, Dimension, ChildSeeds, SimulationNumber, CopulaParameter) 
        endtestdatagen = time.time()
        testdatagentime =float(endtestdatagen-starttestdatagen)
        print("testdatageneration time:", testdatagentime )
    return FullData, RegressionData, TestData, RegressionKDEBandwidth ,RegressionKDERunTime, RegressionDataFullDataMethod #Return the full generated data sample, the data sample prepared for use for the regression part and the test data

def GenerateInitData(Dimension, NumberOfPoints, Index, ChildSeeds,Model,f, BoundGBM,ConditionalModel, CopulaParameter):
    """
    Function to generate data used for setting the bandwidth constant for each model. This data is (pseudo)-independent from the data and the testdata.
    """
    if Model == 'NaiveBayesGBM':
        InitData = NaiveBayesGBMDensity(NumberOfPoints,Dimension,ChildSeeds, False, Index, f, True, BoundGBM,ConditionalModel)
    elif Model == 'BayesBinaryTree': 
        InitData = BayesBinaryTreeDensity(NumberOfPoints,Dimension,ChildSeeds, False, Index, f, True, BoundGBM,ConditionalModel)
    elif Model == 'Copula': 
        InitData = CopulaDensity(NumberOfPoints,Dimension,ChildSeeds, False, Index,  True,CopulaParameter) 
    return InitData

####################
#KERNEL
####################
#The construction of the Kernel estimate should come here


def KernelEstimationRegression(EvaluationDatapoints, InputDataPoints, NumberOfInputPoints,Dimension, BandwidthConstantRegression):
    """
    Function that uses the KernelDensity function from sklearn to generate a kernel density estimator for the combined approaches. Used kernel is the d-dimensional epanechnikov. 
    """
    KernelData = pd.DataFrame(columns=['Estimate'], dtype='float64')
    Bandwidth = (BandwidthConstantRegression)*((log(NumberOfInputPoints)/NumberOfInputPoints)**(1/Dimension))
    KernelModel = KernelDensity(kernel='epanechnikov', bandwidth = Bandwidth).fit(InputDataPoints.to_numpy())
    KernelData = pd.DataFrame( {'Estimate': exp(KernelModel.score_samples(EvaluationDatapoints.to_numpy()))})
    return KernelData, Bandwidth


def KernelEstimationProcedureKDEMethod(FullData, Dimension, Smoothness, NumberOfPoints, TestDataPoints, BandwidthConstant):
    """
    Function that uses the KernelDensity function from sklearn to generate a kernel density estimator for the pure kde method used as a baseline for comparison. Used kernel is the d-dimensional epanechnikov. 
    """
    RunTime = 0
    Bandwidth = (BandwidthConstant)*((1/NumberOfPoints)**(1/(2*Smoothness+Dimension)))
    start = time.time()
    KernelModel = KernelDensity(kernel='epanechnikov', bandwidth = Bandwidth).fit(FullData.to_numpy())
    KernelData = pd.DataFrame( {'Estimate': exp(KernelModel.score_samples(TestDataPoints.to_numpy()))})
    end = time.time()
    RunTime = float(end-start)
    return KernelData, Bandwidth, RunTime


####################
#NETWORK
####################

#Function to check the bounds of parameters. Used to check if parameters are large and for checking if parameters are (very) close to 0
def CheckParameterBounds(Parameters, Bound):
    """
    Function that counts how many parameters are greater than the given bound in absolute value
    """
    Counter = 0
    for i in range (0, Parameters.size):
        if Parameters[i]> Bound:
            Counter = Counter+1
        elif Parameters[i]< -Bound:
            Counter = Counter+1
    return Counter

#Function to setup the network
def setup_network(Layers, Width, InputDimension, X,Y):
    inputs = tf.keras.Input(shape=(InputDimension,))
    x = tf.keras.layers.Dense(Width, activation='relu', bias_initializer='glorot_uniform', kernel_initializer= 'glorot_uniform')(inputs)
    for layer in range(Layers-1):
        x = tf.keras.layers.Dense(Width, activation='relu', bias_initializer='glorot_uniform', kernel_initializer= 'glorot_uniform', kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4))(x)
    output = tf.keras.layers.Dense(1, activation=None, use_bias=False)(x) #Linear output with no bias in the output layer (This is how this network layer is defined in script)
    network = tf.keras.Model(inputs, output)
    network.compile(optimizer = 'adam', loss= 'MSE')
    network.fit(X,Y, epochs = 1, verbose=0 , batch_size=32) #Go through the data once to pre-train the network
    return network


#Function to construct and train the network
def NetworkTrainingMSE(Layers, Width, InputDimension, Data, NumberOfPoints, Sparsity):
    """
    Function that creates and trains a network using keras/tensorflow
    """
    tf.keras.backend.set_floatx('float64')
    X = Data.iloc[:,:InputDimension]
    Y = Data.iloc[:,InputDimension]
    epochs = 20
    end_step = NumberOfPoints*epochs #Continue the pruning (sparsity) for 20

    base_network = setup_network(Layers, Width, InputDimension, X, Y)
    initial_sparsity = float32(0.25)
    SparsityPercent = float32(1.0-(Sparsity/base_network.count_params())) #Sparsity in keras is defined as the number of zeroes

    pruning_params = {'pruning_schedule': tfmotspar.keras.PolynomialDecay(initial_sparsity=initial_sparsity, final_sparsity=SparsityPercent, begin_step=0, end_step=end_step)}

    pruning_network = tfmotspar.keras.prune_low_magnitude(base_network, **pruning_params)
    pruning_network.compile(optimizer = 'adam', loss= 'MSE')

    callbacks = [tfmotspar.keras.UpdatePruningStep()]

    pruning_network.fit(X,Y, epochs = 400, verbose=0 , batch_size=32, callbacks=callbacks)
    MSETrainingSet = pruning_network.evaluate(X,Y, verbose=0)
    pruning_network.summary()
    return pruning_network, MSETrainingSet, base_network.count_params()

#Function to obtain statistics of the parameters from a trained network.
def NetworkParameterStatistics(network, Layers):
    """
    Function that extracts the information about large and small parameters from a trained network.
    """
    Parameters = zeros(0)
    k=1 #Counter to count the layers, used to check if we are in the output layer
    for layer in network.layers:
        if 'input' not in layer.name: #Check that we are not the input layer as that layer does not have any parameters
            if k == Layers: #Check if we are not the output layer, as that layer has no bias parameter
                ParameterAid = layer.get_weights()
                ParameterValue = concatenate((ParameterAid[0],ParameterAid[1]) ,axis=None)
            else:
                ParameterAid = layer.get_weights()
                ParameterValue = ParameterAid[0]
            Parameters = concatenate((Parameters, ParameterValue), axis=None)
            k= k+1
    LargeParameterCount = CheckParameterBounds(Parameters, 1.0)
    DoubleLargeParameterCount = CheckParameterBounds(Parameters, 2.0) 
    #Function to check how many parameters are very close to 0 and how does this relate to the sparsity in the network assumption. 
    NonSmallParameterCount = CheckParameterBounds(Parameters, 0.001)
    NonVerySmallParameterCount = CheckParameterBounds(Parameters, 0.0001)
    return LargeParameterCount, DoubleLargeParameterCount ,NonSmallParameterCount, NonVerySmallParameterCount

def NetworkTrainingProcedure(Layers, Width, InputDimension, Data, NumberOfRepeats, TestPoints, TestValues, NumberOfPoints, Sparsity):
    """
    Function that starts the network training of NumberOfRepeats networks and returns the relevant results from this training.
    """
    start = time.time()
    MSE = zeros(NumberOfRepeats)
    MSETrainingSet = zeros(NumberOfRepeats)
    LargeParameterCount = zeros(NumberOfRepeats)
    DoubleLargeParameterCount = zeros(NumberOfRepeats)
    NonSmallParameterCount = zeros(NumberOfRepeats)
    NonVerySmallParameterCount = zeros(NumberOfRepeats)
    ParameterCount = zeros(NumberOfRepeats)
    #Train the required NumberOfRepeats networks 
    for i in range(0,NumberOfRepeats):
        network, MSETrainingSet[i], ParameterCount[i] = NetworkTrainingMSE(Layers, Width, InputDimension, Data, NumberOfPoints, Sparsity)
        MSE[i] = network.evaluate(TestPoints,TestValues)
        LargeParameterCount[i], DoubleLargeParameterCount[i] ,NonSmallParameterCount[i], NonVerySmallParameterCount[i]= NetworkParameterStatistics(network, Layers)
    NetworkStatistics = pd.DataFrame({'MSE': MSE[:], 'MSETrainingSet': MSETrainingSet[:] ,'LargeParameterCount': LargeParameterCount[:],'DoubleLargeParameterCount': DoubleLargeParameterCount[:] ,'NonSmallParameterCount': NonSmallParameterCount[:], 'NonVerySmallParameterCount': NonVerySmallParameterCount[:] ,'ParameterCount': ParameterCount[:]})
    end = time.time()
    RunTime= float(end-start)
    return NetworkStatistics, RunTime #Return the statistics of the networks

####################
#CALCULATIONS
####################
#Functions that do basic calculation should be placed here
def AltScore(estimator, X):
    """
    Scoring function to get rid of the -inf values in the grid-search by replacing them by something still very negative but finite
    """
    Score = estimator.score_samples(X)
    Score[Score == float('-inf')] = -1e10
    return average(Score)

def MSECalculation(PredictedValues, TrueValues, SampleSize):
    """
    Function to calculate the MSE. Used for all MSE calculations except the networks, which use the network.evaluate method from keras/tensorflow.
    """
    mse = 0.0
    for i in range(0,SampleSize):
        mse = mse + (TrueValues.iloc[i,0]-PredictedValues.iloc[i,0])**2
    mse = mse/SampleSize
    return mse

#Function to calculate the bound on the generated GBM functions
def BoundCalculation(f, Steps):
    """
    Function to find the maximum of the generated GBM.
    """
    EstimatedMax = 0.0 #Function will be used to estimate the max of a density (used for setting the rejection sampling bound) so max will be >0
    Points = 2*Steps
    for i in range(0,Points+1):
        Aid = f(i/(Points+1))
        if Aid >= EstimatedMax:
            EstimatedMax = Aid
    Bound = EstimatedMax + 0.1 #Add a margin for safety
    return Bound
    #We do not need to calculate the bounds for the conditional distributions based on the GBMs as they are the same as the marginal version (mixing with a mirrored version and shifting do not affect the bound) 

####################
#SAVE AND IMPORT
####################

#Function to store the results of the simulation
def SaveReport(Model, ConditionalModel, SimulationNumber, Seed, NumberOfPoints,TestPoints, Smoothness, Dimension ,KDERunTime, NetworkRunTime,  Repeats,layers,width, KDEMSE, NetworkStatistics, NetworkStatisticsFullDataMethod, mseKDERegression, mseKDERegressionFullData,  mseZeroEstimator, KDEOptimalBandwidth, RegressionKDEBandwidth, RegressionKDEBandwidthFullData, RegressionKDERunTime, BandwidthConstant, BandwidthConstantRegression, BrownianMotionSteps, SparsityTA, SparsityFA):
    """
    This function collects all the information, puts it in a pandas dataframe and writes it to a .csv file.
    """
    #Get the relevant data from the NetworkStatistics DataFrame
    ParameterCount = NetworkStatistics.iloc[0,6] #Parameter count is the same across all the networks
    AverageNetworkMSE = average(NetworkStatistics.loc[:,'MSE'].values)
    AverageNetworkNonSmallParameters =  average(NetworkStatistics.loc[:,'NonSmallParameterCount'].values)
    AverageNetworkNonVerySmallParameters =  average(NetworkStatistics.loc[:,'NonVerySmallParameterCount'].values)
    AverageNetworkLargeParameters =  average(NetworkStatistics.loc[:,'LargeParameterCount'].values)
    AverageNetworkVeryLargeParameters =  average(NetworkStatistics.loc[:,'DoubleLargeParameterCount'].values)
    NetworkStatistics.sort_values(by =['MSE'], inplace=True)
    NetworkMSEData = NetworkStatistics['MSE']
    QuantileMSE = percentile(NetworkMSEData.to_numpy(), [0,25,50,75,100])
    BestNetworkNonSmallParameters = NetworkStatistics.iloc[0,4]
    BestNetworkNonVerySmallParameters = NetworkStatistics.iloc[0,5]
    BestNetworkLargeParameters = NetworkStatistics.iloc[0,2]
    BestNetworkVeryLargeParameters = NetworkStatistics.iloc[0,3]
    BestNetworkTrainingError = NetworkStatistics.iloc[0,1]
    NetworkStatistics.sort_values(by =['MSETrainingSet'], inplace=True)
    BestTrainingNetworkMSE = NetworkStatistics.iloc[0,0]
    BestTrainingNetworkNonSmallParameters = NetworkStatistics.iloc[0,4]
    BestTrainingNetworkNonVerySmallParameters = NetworkStatistics.iloc[0,5]
    BestTrainingNetworkLargeParameters = NetworkStatistics.iloc[0,2]
    BestTrainingNetworkVeryLargeParameters = NetworkStatistics.iloc[0,3]
    BestTrainingNetworkTrainingError = NetworkStatistics.iloc[0,1]


    #Get the relevant data from the NetworkStatisticsFullDataMethod
    AverageNetworkMSEFullDataMethod = average(NetworkStatisticsFullDataMethod.loc[:,'MSE'].values)
    AverageNetworkNonSmallParametersFullDataMethod =  average(NetworkStatisticsFullDataMethod.loc[:,'NonSmallParameterCount'].values)
    AverageNetworkNonVerySmallParametersFullDataMethod =  average(NetworkStatisticsFullDataMethod.loc[:,'NonVerySmallParameterCount'].values)
    AverageNetworkLargeParametersFullDataMethod =  average(NetworkStatisticsFullDataMethod.loc[:,'LargeParameterCount'].values)
    AverageNetworkVeryLargeParametersFullDataMethod =  average(NetworkStatisticsFullDataMethod.loc[:,'DoubleLargeParameterCount'].values)
    NetworkStatisticsFullDataMethod.sort_values(by =['MSE'], inplace=True)
    NetworkMSEDataFullDataMethod = NetworkStatisticsFullDataMethod['MSE']
    QuantileMSEFullDataMethod = percentile(NetworkMSEDataFullDataMethod.to_numpy(), [0,25,50,75,100])    
    BestNetworkNonSmallParametersFullDataMethod = NetworkStatisticsFullDataMethod.iloc[0,4]
    BestNetworkNonVerySmallParametersFullDataMethod = NetworkStatisticsFullDataMethod.iloc[0,5]
    BestNetworkLargeParametersFullDataMethod = NetworkStatisticsFullDataMethod.iloc[0,2]
    BestNetworkVeryLargeParametersFullDataMethod = NetworkStatisticsFullDataMethod.iloc[0,3]
    BestNetworkTrainingErrorFA = NetworkStatisticsFullDataMethod.iloc[0,1]
    NetworkStatisticsFullDataMethod.sort_values(by =['MSETrainingSet'], inplace=True)
    BestTrainingNetworkMSEFullDataMethod = NetworkStatisticsFullDataMethod.iloc[0,0]
    BestTrainingNetworkNonSmallParametersFullDataMethod= NetworkStatisticsFullDataMethod.iloc[0,4]
    BestTrainingNetworkNonVerySmallParametersFullDataMethod = NetworkStatisticsFullDataMethod.iloc[0,5]
    BestTrainingNetworkLargeParametersFullDataMethod = NetworkStatisticsFullDataMethod.iloc[0,2]
    BestTrainingNetworkVeryLargeParametersFullDataMethod = NetworkStatisticsFullDataMethod.iloc[0,3]
    BestTrainingNetworkTrainingErrorFA = NetworkStatisticsFullDataMethod.iloc[0,1]


    Report = pd.DataFrame({'Model':[Model], 'ConditionalModel':[ConditionalModel],'SimulationNumber': [SimulationNumber], 'Seed': [Seed], 'n': [NumberOfPoints], 'BrownianMotionSteps': [BrownianMotionSteps], 
                             'TestPoints': [TestPoints], 'Smoothness': [Smoothness], 'Dimension': [Dimension] ,'KDERunTime': [KDERunTime], 'RegressionKDERunTime': [RegressionKDERunTime],
                             'BandwidthConstant': [BandwidthConstant], 'KDEOptimalBandwidth': [KDEOptimalBandwidth],
                             'BandwidthConstantRegression': [BandwidthConstantRegression],'RegressionKDEBandwidth': [RegressionKDEBandwidth],'RegressionKDEBandwidthFullData':[RegressionKDEBandwidthFullData],
                             'NetworkRunTime': [NetworkRunTime], 'NetworkRepeats': [Repeats], 'layers': [layers], 'width':[width], 'Parameters': [ParameterCount], 'NonzeroWeightsTA':[SparsityTA], 'NonzeroWeightsFA': [SparsityFA],
                             'MSEZeroEstimator': [mseZeroEstimator], 'KDEMSE':[KDEMSE],'mseKDERegression': [mseKDERegression] , 'mseKDERegressionFullData': [mseKDERegressionFullData] ,
                             
                             'BestTrainingNetworkMSE': [BestTrainingNetworkMSE], 'AverageNetworkMSE': [AverageNetworkMSE],'BestNetworkTrainingError':[BestNetworkTrainingError]  ,'BestTrainingNetworkTrainingError': [BestTrainingNetworkTrainingError],
                             'MinimumMSE': [QuantileMSE[0]], 'FirstQuantile': [QuantileMSE[1]], 'SecondQuantile': [QuantileMSE[2]], 'ThirdQuantile': [QuantileMSE[3]], 'MaximumMSE': [QuantileMSE[4]], 
                             'BestTrainingNetworkMSEFullDataMethod': [BestTrainingNetworkMSEFullDataMethod], 'AverageNetworkMSEFullDataMethod': [AverageNetworkMSEFullDataMethod], 'BestNetworkTrainingErrorFA':[BestNetworkTrainingErrorFA]  ,'BestTrainingNetworkTrainingErrorFA': [BestTrainingNetworkTrainingErrorFA],
                             'MinimumMSEFullDataMethod': [QuantileMSEFullDataMethod[0]], 'FirstQuantileFullDataMethod': [QuantileMSEFullDataMethod[1]], 'SecondQuantileFullDataMethod': [QuantileMSEFullDataMethod[2]], 'ThirdQuantileFullDataMethod': [QuantileMSEFullDataMethod[3]], 'MaximumMSEFullDataMethod': [QuantileMSEFullDataMethod[4]], 
                               
                             'BestNetworkParameters>0.01': [BestNetworkNonSmallParameters], 'BestNetworkParameters>0.0001': [BestNetworkNonVerySmallParameters], 'BestNetworkParameters>1': [BestNetworkLargeParameters],
                             'BestNetworkParameters>2': [BestNetworkVeryLargeParameters], 'BestTrainingNetworkParameters>0.01': [BestTrainingNetworkNonSmallParameters],
                             'BestTrainingNetworkParameters>0.0001': [BestTrainingNetworkNonVerySmallParameters], 'BestTrainingNetworkParameters>1': [BestTrainingNetworkLargeParameters],
                             'BestTrainingNetworkParameters>2': [BestTrainingNetworkVeryLargeParameters], 'AverageNetworkParameters>0.01': [AverageNetworkNonSmallParameters],
                             'AverageNetworkParameters>0.0001': [AverageNetworkNonVerySmallParameters], 'AverageNetworkParameters>1': [AverageNetworkLargeParameters],
                             'AverageNetworkParameters>2': [AverageNetworkVeryLargeParameters] ,

                             'BestNetworkFullDataMethodParameters>0.01': [BestNetworkNonSmallParametersFullDataMethod], 'BestNetworkFullDataMethodParameters>0.0001': [BestNetworkNonVerySmallParametersFullDataMethod], 'BestNetworkFullDataMethodParameters>1': [BestNetworkLargeParametersFullDataMethod],
                             'BestNetworkFullDataMethodParameters>2': [BestNetworkVeryLargeParametersFullDataMethod], 'BestTrainingNetworkFullDataMethodParameters>0.01': [BestTrainingNetworkNonSmallParametersFullDataMethod],
                             'BestTrainingNetworkFullDataMethodParameters>0.0001': [BestTrainingNetworkNonVerySmallParametersFullDataMethod], 'BestTrainingNetworkFullDataMethodParameters>1': [BestTrainingNetworkLargeParametersFullDataMethod],
                             'BestTrainingNetworkFullDataMethodParameters>2': [BestTrainingNetworkVeryLargeParametersFullDataMethod], 'AverageNetworkFullDataMethodParameters>0.01': [AverageNetworkNonSmallParametersFullDataMethod],
                             'AverageNetworkFullDataMethodParameters>0.0001': [AverageNetworkNonVerySmallParametersFullDataMethod], 'AverageNetworkFullDataMethodParameters>1': [AverageNetworkLargeParametersFullDataMethod],
                             'AverageNetworkFullDataMethodParameters>2': [AverageNetworkVeryLargeParametersFullDataMethod],

                             
                             'Timestamp': [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())] })
    timemark = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    Report.to_csv("./outputdata/SimulationReport{}.csv".format(timemark),index=False, encoding='utf8') 
    NetworkStatistics.to_csv("./outputdata/SimulationNetworkSplitData{}.csv".format(timemark),index=False, encoding='utf8') 
    NetworkStatisticsFullDataMethod.to_csv("./outputdata/SimulationNetworkFullData{}.csv".format(timemark),index=False, encoding='utf8') 

####################
#INITIALIZATION FUNCTION
####################

def InitGBM(ChildSeeds, Steps, Initial, Dimension, ConditionalModel):
    """
    Function that creates the Geometric Brownian Motion path functions.
    """
    index = int(0)
    GBMPath = GBMPathGeneration(Steps, Initial, ChildSeeds[index])
    f1 = GBMDensity(GBMPath,Steps) 
    f = array([f1])
    Bound1 = BoundCalculation(f1, Steps)
    Bound = array([Bound1])
    for d in range (1, Dimension):
        if d%3 > 0: #Use inequality here to allow to increase the number of nearunif densities that we use (if we change to == then we can decrease the number of nearunif densities we use)
            faid = NearUnifDensity(Dimension)
        else:
            GBMPath = GBMPathGeneration(Steps, Initial, ChildSeeds[index+d])
            if ConditionalModel == 'Mixing':
                faid = GBMDensity(GBMPath,Steps) 
            elif ConditionalModel == 'Shifting':
                faid = GBMNormalizedConcentratedVariationDensity(GBMPath,Steps) 
        f = append(f,faid)
        Boundaid = BoundCalculation(faid, Steps)
        Bound = append(Bound, Boundaid)
    print(Bound)
    return f, Bound

#Function to initiliaze the bandwidthconstant. We take the average of the cross-validation result on 5 data-sets of 200 points each (as 200 points is the minimum n we plan to run the simulation with)
def InitializeBandwidth(Model, Dimension, Smoothness, ChildSeeds, f, BoundGBM, ConditionalModel, CopulaParameter):
    """
    Function to determine the bandwidth constant for each model. It takes the average of 5 cross-validation procedures on 5 newly generated data-sets of size 200.
    """
    bandwidths = arange(0.05, 1.1, .005)   
    BandwidthOptimal = ones(5)
    for i in range(1,6):
        InitData= GenerateInitData(Dimension, 200, i,  ChildSeeds, Model, f,  BoundGBM, ConditionalModel, CopulaParameter)
        grid = GridSearchCV(KernelDensity(kernel='epanechnikov'),{'bandwidth': bandwidths}, scoring=AltScore, cv=50)
        grid.fit(InitData)
        BandwidthOptimal[i-1] = grid.best_estimator_.bandwidth
        print(BandwidthOptimal[i-1])
    BandWidthOpt = average(BandwidthOptimal)
    print("Bandwidths:", BandwidthOptimal)
    BandwidthConstant =  BandWidthOpt/((1/200)**(1/(2*Smoothness+Dimension)))
    BandwidthConstantRegression = (BandWidthOpt/((log(200)/200)**(1/(Dimension))) )
    return BandwidthConstant, BandwidthConstantRegression

####################
#MAIN FUNCTION
####################

def Mainsimulation():
    """
    The main function. The definitions of which model and other constants are defined here.
    """
    
    #Specify which model to use. Should be one of: 'NaiveBayesGBM', 'BayesBinaryTree', 'Copula' 
    Model = 'BayesBinaryTree' 
    #Specify which conditional densities to use. Should be one of: 'Mixing', 'Shifting'. This has no effect on the Copula model
    ConditionalModel = 'Shifting'
    #Defintion of the constants
    n = 200 #Keep even at all times, as this quantity corresponds to 2n in the script.
    TestDataNumber = 100000 #Number of test data points, these are only used for checking the perfomance of the methods
    Dimension = 4 #Dimension of the input. Code as allows for dimensions from 2 to (and including) 24 (dimension 1 should work also)
    Smoothness = 0.5 #The Copula model is defined for 0.5. A true GBM-path has smoothness <0.5 
    
    layers = max(1,int(ceil(log(n)/log(2)))) #Number of layers in the DNN
    width = min(1000,int(ceil(n**(1/(2*Smoothness+1))))) #Number of neurons in each layer of the DNN, the width is capped at 1000 to prevent the networks to become to large for the memory  
    
    SparsityTA = 2*(n/2)*log(n/2)*((n/2)**(-(2*Smoothness)/(2*Smoothness+1))) #Added the multiplicative constant 2 to make it less likely that the network is too sparse
    SparsityFA = (2*n)*log(n)*(n**(-(2*Smoothness)/(2*Smoothness+1))) #Added the multiplicative constant 2 to make it less likely that the network is too sparse
    SimulationNumber = 1 #Number to keep track which simulation of the data set this is. This number can be 1,2,3,4,5.
    NumberOfRepeats = 50 #Number of networks per networkmethod that we train on the data

    #Number of time steps in the generation of the Brownian Motion path, set to 2^(k) for some integer k (so that together with the initial value we have 2^k+1 points) so we can use the integrate.romb functio
    k = 21
    BrownianMotionSteps = int(2**k) 

    Initial = 0 #Initial value for the BM-path generation
    
    #Parameter for the CopulaFamily
    CopulaParameter = ones(Dimension-1) #We need one binary copula less than the number of dimensions we have
    for i in range (0,Dimension-1):
        CopulaParameter[i]= -1+(2*i)/(Dimension-2)
        if CopulaParameter[i] == 0.0:
            CopulaParameter[i] = 0.01 #if we the parameter is euqal to zero we get the independence copula which we avoid in this way

    # get the version of TensorFlow
    print("TensorFlow version: {}".format(tf.__version__))

    # Check that TensorFlow was build with CUDA to use the gpus
    print("Device name: {}".format(tf.test.gpu_device_name()))
    print("Build with GPU Support? {}".format(tf.test.is_built_with_gpu_support()))
    print("Build with CUDA? {} ".format(tf.test.is_built_with_cuda()))

    #Create sequences of seeds from one seed (to make the results reproducible and to ensure that the (pseudo)-random generation in various functions is not too related)
    Seed = 15252
    SequenceSeeder = random.SeedSequence(Seed)
    ChildSeeds = SequenceSeeder.spawn(5000) #The childseeds are used in the following way: 1000 seeds are reserved for each model. For each simulationnumber there are then 100 seeds. The first 50 for generating the test data, the last 50 for generating the (training) data. The second 500 of each 1000 are used for initialising the bandwithconstant. The first 25 seeds are for the generation of the brownian motion path.
    #Seeds given away so far:
    #0-24 GBM Paths
    #1000-1999 Copula Model 
    #2000-2999 NaiveBayesGBM Model
    #3000-3999 BayesBinaryTree Model 


    #Initilization functions
    start_time = time.time()
    if Model == 'Copula':
        f = zeros(Dimension) #We do not use the GBM paths (and their bounds) in the copula model, so we do not need to initialize them.
        BoundGBM = ones(Dimension)
    else:
        f, BoundGBM = InitGBM(ChildSeeds, BrownianMotionSteps, Initial, Dimension, ConditionalModel)
    BandwidthConstant, BandwidthConstantRegression = InitializeBandwidth(Model, Dimension, Smoothness, ChildSeeds, f, BoundGBM, ConditionalModel, CopulaParameter)

    #Create data
    FullData, RegressionData, TestData, RegressionKDEBandwidth, RegressionKDERunTime, RegressionDataFullDataMethod= GenerateData(Dimension, n, SimulationNumber, ChildSeeds, TestDataNumber, f, Model, BandwidthConstantRegression, BoundGBM,ConditionalModel,CopulaParameter) 
    #Store the testpoints without the density values, used by KDE and network to generate estimations of the density (and to compare those with the true density value)
    TestPoints = TestData.drop(labels = 'Z',axis = 'columns')
    TestValues = TestData["Z"]
    TestValues =TestValues.to_frame()



    #Get the MSE for the KDE used for the regression
    splitvalue = int(n/2)
    X = FullData.iloc[:splitvalue,:]
    KDERegression, RegressionKDEBandwidthControlVersion = KernelEstimationRegression(TestPoints,X, splitvalue, Dimension, BandwidthConstantRegression )
    mseKDERegression = MSECalculation(KDERegression,TestValues,TestDataNumber)
    KDERegressionFullData, RegressionKDEBandwidthFullData = KernelEstimationRegression(TestPoints,FullData, n, Dimension, BandwidthConstantRegression)
    mseKDERegressionFullData = MSECalculation(KDERegressionFullData,TestValues,TestDataNumber)   
    print("Data-generation and kernel estimation running time: {0:1f}s".format(time.time()-start_time))
    #Train the networks
    #Theoretical method
    NetworkStatistics, NetworkRunTime = NetworkTrainingProcedure(layers,width,Dimension,RegressionData,NumberOfRepeats ,TestPoints,TestValues, int(n/2), SparsityTA)
    #Full data method
    NetworkStatisticsFullDataMethod, NetworkRunTimeFullDataMethod = NetworkTrainingProcedure(layers,width,Dimension,RegressionDataFullDataMethod,NumberOfRepeats ,TestPoints,TestValues, n, SparsityFA)


    #Calculate the KDE with theoretical optimal bandwidth
    KDEtest, KDEOptimalBandwidth, KDERunTime = KernelEstimationProcedureKDEMethod(FullData, Dimension, Smoothness,n, TestPoints,  BandwidthConstant)

    #Calculate the MSE based on the Testdata for the KDE with optimal bandwidth
    mseKernel = MSECalculation(KDEtest,TestValues,TestDataNumber)
    
    #Score the MSE of the estimator that always give back zero to give a baseline to compare with
    ZeroEstimator = pd.DataFrame({'Z': zeros(TestDataNumber)})
    mseZeroEstimate = MSECalculation(ZeroEstimator, TestValues, TestDataNumber)

    #Store the results in a file
    SaveReport(Model, ConditionalModel, SimulationNumber, Seed, n, TestDataNumber, Smoothness, Dimension ,KDERunTime, NetworkRunTime, NumberOfRepeats, layers, width,mseKernel,NetworkStatistics, NetworkStatisticsFullDataMethod, mseKDERegression, mseKDERegressionFullData ,mseZeroEstimate, KDEOptimalBandwidth, RegressionKDEBandwidth, RegressionKDEBandwidthFullData ,  RegressionKDERunTime, BandwidthConstant, BandwidthConstantRegression, BrownianMotionSteps, SparsityTA, SparsityFA)
    print("Total running time: {0:1f}s".format(time.time()-start_time))
    
####################
#RUN THE PROGRAM
####################
Mainsimulation()
