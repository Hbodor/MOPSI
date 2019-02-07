import numpy as np
from math import sqrt,exp

def prix_recursif_us(x,k,N,gain):
    res = p * prix_recursif_us(x*u,k+1,N) + (1-p)* prix_recursif_am(x*d,k+1,N)
    return max(res/(1+r),gain(x))
        
def gain_put(x,K):
    return max(K-x,0)
    
def gain_call(x,K):
    return max(x-K,0)


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
    

    
T = 5 #number of years
    
sigma=0.3 # Volatility 
r_0=0.03 # interest rate (per year)

K=20 #Strike price
x_0=20 #Initial price

N=10000 #Number of steps
k= 5 # Exercice times  (frequency ? ) in T ( bermudean option )  : eg g every T/12 ; k = N if american

r=T*r_0/N

a=sigma*sqrt(T/N) 

d=exp(-a)*(1+r) # down factor
u=exp(a)*(1+r) # up factor 

p= (1+r-d)/(u-d)

n_k=N//k #we scale the exercice times to the number of steps
N = n_k * k #we round the steps


U=prix_am(x_0,N,n_k,gain_put)

print(U)