from numpy import linalg, zeros, ones, hstack, asarray
import itertools


def basis_vector(n, i):
    """ Return an array like [0, 0, ..., 1, ..., 0, 0]

    >>> from multipolyfit.core import basis_vector
    >>> basis_vector(3, 1)
    array([0, 1, 0])
    >>> basis_vector(5, 4)
    array([0, 0, 0, 0, 1])
    """
    x = zeros(n, dtype=int)
    x[i] = 1
    return x

def as_tall(x):
    """ Turns a row vector into a column vector """
    return x.reshape(x.shape + (1,))

def multipolyfit(xs, y, deg, full=False, model_out=True, powers_out=False):
    """
    Least squares multivariate polynomial fit

    Fit a polynomial like ``y = a**2 + 3a - 2ab + 4b**2 - 1``
    with many covariates a, b, c, ...

    Parameters
    ----------

    xs : array_like, shape (M, k)
         x-coordinates of the k covariates over the M sample points
    y :  array_like, shape(M,)
         y-coordinates of the sample points.
    deg : int
         Degree o fthe fitting polynomial
    model_out : bool (defaults to True)
         If True return a callable function
         If False return an array of coefficients
    powers_out : bool (defaults to False)
         Returns the meaning of each of the coefficients in the form of an
         iterator that gives the powers over the inputs and 1
         For example if xs corresponds to the covariates a,b,c then the array
         [1, 2, 1, 0] corresponds to 1**1 * a**2 * b**1 * c**0

    See Also
    --------
        numpy.polyfit

    """
    y = asarray(y).squeeze()
    rows = y.shape[0]
    xs = asarray(xs)
    num_covariates = xs.shape[1]
    xs = hstack((ones((xs.shape[0], 1), dtype=xs.dtype) , xs))

    generators = [basis_vector(num_covariates+1, i)
                            for i in range(num_covariates+1)]

    # All combinations of degrees
    powers = list(map(sum, itertools.combinations_with_replacement(generators, deg)))
   

    # Raise data to specified degree pattern, stack in order
    A = hstack(asarray([as_tall((xs**p).prod(1)) for p in powers]))

    beta = linalg.lstsq(A, y)[0]
    if model_out:
        return mk_model(beta, powers)

    if powers_out:
        return beta, powers
    return beta

def mk_model(beta, powers):
    """ Create a callable pyTaun function out of beta/powers from multipolyfit

    This function is callable from within multipolyfit using the model_out flag
    """

    # Create a function that takes in many x values
    # and returns an approximate y value
    def model(*args):
        

        num_covariates = len(powers[0]) - 1
        args=args[0]
        if len(args)!=(num_covariates):
            raise ValueError("Expected %d inputs"%num_covariates)
        xs = [1]+args
        return sum([coeff * (xs**p).prod()
                             for p, coeff in zip(powers, beta)])
    return model

def mk_sympy_function(beta, powers):
    from sympy import symbols, Add, Mul, S

    num_covariates = len(powers[0]) - 1
    xs = (S.One,) + symbols('x0:%d'%num_covariates)
    return Add(*[coeff * Mul(*[x**deg for x, deg in zip(xs, power)])
                        for power, coeff in zip(powers, beta)])
          

##########################################################################################################################################################################################################
#Neural Network

from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

class ApproxNet:
    @staticmethod
    def build(input_length):
        # initialize the model
        model = Sequential()
        
        #input layer
        model.add(Dense(input_length*3 , input_dim=input_length,kernel_initializer='he_normal',activation="selu"))
        
        #hidden layers
        model.add(Dense(input_length*2, activation="selu"))
        
        #hidden layers
        model.add(Dense(input_length, activation="selu"))
        
        #output layer
        model.add(Dense(1,activation="selu"))

        model.compile(loss='mse', optimizer='RMSprop', metrics=['accuracy','mse', 'mae', 'mape', 'cosine'])
        # return the constructed network architecture
        return model
        
def approximation(j,Tau,paths):
    
    model = ApproxNet.build(len(paths[0][0]))
    
    payOffs=[]
    x=np.zeros((len(paths),len(paths[0][0])))

    for i,path in enumerate(paths):
        payOffs.append(B(j,Tau[i][j+1])*payOff(path[Tau[i][j+1]],K))
        x[i,:]=np.asarray(path[j])
    # x = np.log(x)
    # print(np.mean(x))
    
    print(x.shape,x[0:1].shape)
    
    history = model.fit(x, payOffs, batch_size=32, epochs=100, verbose=0)
    

    
    return (model,history)


####################################
import numpy as np
from scipy.interpolate import *
#import multipolyfit

#Parameters


sigma=0.3 #Volatility

M=5000 #Number of paths

T=1 # Expiration (in years)
dt = 1/3 # steps of exercise (in years)

N = int(T/dt) #number of iterations


r=0.03 #Risk-free interest rate (annual)


X_0=[20]
K=X_0

reg=True # regression or neural network 

############################################################################################################



#Genaerating a p-dimensional Brownian motion

from scipy.stats import norm
# import matplotlib.pyplot as plt
from math import sqrt,log,exp

# Process parameters



def genBM():
    W=[]
    for i in range(len(X_0)):
        # Initial condition.
        x = 0
        
        # Number of iterations to compute.
        n = N
        
        
        # Iterate to compute the steps of the Brownian motion.
        
        D=[]
        X=[]
        
        for k in range(n):
            x = x + norm.rvs(scale=sqrt(dt))
            D.append(x)
        W.append(D)
    return W



Sigma=sigma*np.eye(len(X_0))

    #since sum(sigma(i,j)**2) will be calculated a lot of times, we will save iy in variable to reduce time complexity
SIGMA2=[]

for i in range(len(Sigma[0])):
    summ=0
    for j in range( len(Sigma[0]) ):
        summ+=Sigma[i][j]**2
    SIGMA2.append(summ)

def next(X,k,W,dt):
    L=[]
    for i in range(len(X)):
        summ=0  #will play the role of sum(sigma(i,j)*Wk(j))
        for j in range( len(X)):
            summ+=Sigma[i][j]*W[j][k-1]  #Because we willl never calculate the first value of the assets
        exposant=(r-1/2*SIGMA2[i])*k*dt+summ
        L.append(X[i]*exp(exposant))
    return L
        







#########################################################################################################################################################################################################"


#function that generates a path
def pathGen(X_0):
    W=genBM()
    L=[0 for i in range(N+1)]
    L[0]=X_0
    for i in range(1,N+1):
        L[i]=next(X_0,i,W,dt)
    return L






##try it 
#print(pathGen(X_0))

#Gain function
def payOff(X,K):
    return max(np.sum(K)-np.sum(X),0)




#Interest rate function
def B(j,k):
    # P=1
    # for i in range(j,k):
    #     P*=1/(1+r)
    return exp(-r*(k-j)*dt)

def price(Taus,paths):
    Q=0
    M=len(Taus)
    for m in range(M):

        Q+=B(0,Taus[m][0])*payOff(paths[m][Taus[m][0]],K)
    Q=Q/M
    return Q
    
#regression to find the polynom
def regression(j,Tau,paths):
    payOffs=[]
    x=[]
    for (i,path) in enumerate(paths):
        
        payOffs.append(B(j,Tau[i][j+1])*payOff(path[Tau[i][j+1]],K))
        
        x.append(path[j])
    p3=multipolyfit(x, payOffs, 5)
    return p3   
HISTORY=[]
HISTORY2=[]
for kk in range(100):
      
    #Generating paths
    paths=[]
    for i in range(M):
        path=pathGen(X_0)
        paths.append(path)
    # 
    
    #Defining Tausm
    Taus=[[0 for i in range(N+1)] for j in range(M)]
    
    for i in range(M):
        Taus[i][-1]=N
    
    
    
    #construction of Taus
    
    if reg:
    #REGRESSION
    
    
        for j in range(N-1,-1,-1):
            
            regresseur=regression(j,Taus,paths)
            
            
            for i in range(M):
                if ( payOff(paths[i][j],K) >= regresseur( paths[i][j] ) ):
                    Taus[i][j]=j
                else : 
                    Taus[i][j]=Taus[i][j+1]
    else:
        
    
    #Neural Network
        
        for j in range(N-1,-1,-1):
            
            model,history = approximation(j,Taus,paths)
            HISTORY.append(history.history["loss"])
            HISTORY2.append(history)
            for i in range(M):
                
                if ( payOff(paths[i][j],K) >= model.predict( np.resize(paths[i][j],(1,len(X_0))) ) ):
                    Taus[i][j]=j
                else : 
                    Taus[i][j]=Taus[i][j+1]
    
    
    
    print(price(Taus,paths))









            
    
    