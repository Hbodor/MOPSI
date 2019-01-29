import numpy as np
from math import sqrt

import itertools

def prix_recursif_us(x,k,N,gain):
    res = p * prix_recursif_us(x*u,k+1,N) + (1-p)* prix_recursif_am(x*d,k+1,N)
    return max(res/(1+r),gain(x))
        
def gain_put(x,K):
    return max(K-x,0)
    
def gain_call(x,K):
    return max(x-K,0)

def payoff(X,K):
    return np.sum([gain_put(x,K[i]) for (i,x) in enumerate(X)])

def prix_am(x_0,N,k,gain):
    
    U=np.zeros((N+1,N+1))
    
    x=[x_0*u**i * d**(N-i) for i in range(N+1)] 

    U[N,0:N+1] = [gain(t,K) for t in x]
    
    for n in range(N-1,-1,-1):
        U[n,0:n+1] = U[n+1,1:n+2].dot(p) + U[n+1,0:n+1].dot(1-p)
        
        U[n,0:n+1] = U[n,0:n+1].dot(1/(1+r))
        
        x=[x_0*u**i * d**(n-i) for i in range(n+1)]
        
        if(n%k==0):
            U[n,0:n+1] = [max(u,gain(x[k],K)) for (k,u) in enumerate(U[n,0:n+1])]
        
    return U[0,0]
    
    
def prix_vect_am(X_0,K,N,gain):
    
    

    set=[X_0[0]*u**i * d**(N-i) for i in range(N+1)] 
    for k in range(1,len(X_0)):
        set=list(itertools.product(set,[X_0[k]*u**i * d**(N-i) for i in range(N+1)]))
        
        for (i,s) in enumerate(set):
            set[i]=list(s)
            if type(s) in [tuple,list]:
                if type(s[0])==float:
                    set[i]=[s[0]]+[s[1]]
                else:
                    set[i]=list(s[0])+[s[1]]
    
    
    U=np.zeros((N+1,len(set)))
    powers=[p**i * (1-p)**(len(X_0)-i) for i in range(len(X_0)+1)]




    U[N,0:N+1] = [gain(t) for t in x]
    
    for n in range(N-1,-1,-1):

        
        set=[X_0[0]*u**i * d**(n-i) for i in range(n+1)] 
        
        
        for k in range(1,len(X_0)):
            set=list(itertools.product(set,[X_0[k]*u**i * d**(n-i) for i in range(n+1)]))
            
            for (i,s) in enumerate(set):
                set[i]=list(s)
                if type(s) in [tuple,list]:
                    if type(s[0])==float:
                        set[i]=[s[0]]+[s[1]]
                    else:
                        set[i]=list(s[0])+[s[1]]
                        
        U[n,0:len(set)] = U[n+1,1:n+2].dot(p) + U[n+1,0:n+1].dot(1-p)
        
        U[n,0:len(set)] = U[n,0:len(set)].dot(1/(1+r))
                            
        U[n,0:len(set)] = [max(u,gain(set[k])) for (k,u) in enumerate(U[n,0:len(set)])]
        
    return U[0,0]
    
sigma=0.3
r_0=0.1
K=100
x_0=100

N=10000
k=10 #1 for classic one, but != 1 for testing convergence to BS model.
r=r_0/N
d=1-sigma/sqrt(N)
u=1+sigma/sqrt(N)
p= (1+r-d)/(u-d)


U=prix_am(x_0,N,k,gain_put)