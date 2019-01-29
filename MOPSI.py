import numpy as np
from math import sqrt

def prix_recursif_us(x,k,N,gain):
    res = p * prix_recursif_us(x*u,k+1,N) + (1-p)* prix_recursif_am(x*d,k+1,N)
    return max(res/(1+r),gain(x))
    
def gain_put(x):
    return max(K-x,0)
    
def gain_call(x):
    return max(x-K,0)


def prix_am(x_0,N,gain):
    
    U=np.zeros((N+1,N+1))
    
    x=[x_0*u**i * d**(N-i) for i in range(N+1)] 

    U[N,0:N+1] = [gain(t) for t in x]
    
    for n in range(N-1,-1,-1):
        U[n,0:n+1] = U[n+1,1:n+2].dot(p) + U[n+1,0:n+1].dot(1-p)
        
        U[n,0:n+1] = U[n,0:n+1].dot(1/(1+r))
        
        x=[x_0*u**i * d**(n-i) for i in range(n+1)] 
        
        U[n,0:n+1] = [max(u,gain(x[k])) for (k,u) in enumerate(U[n,0:n+1])]
        
    return U[0,0]
    
sigma=0.3
r_0=0.1
K=100
x_0=100

N=10
r=r_0/N
d=1-sigma/sqrt(N)
u=1+sigma/sqrt(N)
p= (1+r-d)/(u-d)


U=prix_am(x_0,N,gain_put)