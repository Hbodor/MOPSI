import numpy as np
from math import sqrt,exp

import itertools


class Node:
    def __init__(self,v,p):
        self.v=v;
        self.p=p;
    def __mul__(self,other):
        if type(self.v)==str:
            return Node(self.v+other.v,other.p)
            
        if type(self.v)==list:
            if type(self.v[0])==str:
                return Node( [self.v[i]+other.v[i] for (i,x) in enumerate(self.v)],[other.p[i] for (i,x) in enumerate(self.v)] ) 
            

            return Node( [self.v[i]*other.v[i] for (i,x) in enumerate(self.v)],[other.p[i] for (i,x) in enumerate(self.v)] ) 
            
        return Node(self.v*other.v,other.p)
    def __str__(self):
        return "("+str(self.v)+","+str(self.p)+")"
    def __repr__(self):
        return self.__str__()
        
    def __eq__(self,other):
        
        return self.p==other.p and [str_to_int(node) for node in self.v]== [str_to_int(node) for node in other.v]
        
    def proba(self):
        if type(self.p)!=list:
            return self.p
        pr=1
        for p in self.p:
            pr*=p
        return pr
        
def prix_recursif_us(x,k,N,gain):
    res = p * prix_recursif_us(x*u,k+1,N) + (1-p)* prix_recursif_am(x*d,k+1,N)
    return max(res/(1+r),gain(x))
        
def gain_put(x,K):
    return max(K-x,0)
    
def gain_call(x,K):
    return max(x-K,0)


def prix_am(x_0,K,N,gain):
    
    U=np.zeros((N+1,N+1))
    
    x=[x_0* pow(u,i) * pow(d,(N-i)) for i in range(N+1)] 

    U[N,0:N+1] = [gain(t,K) for t in x]
    
    for n in range(N-1,-1,-1):
        U[n,0:n+1] = U[n+1,1:n+2].dot(p) + U[n+1,0:n+1].dot(1-p)
        
        U[n,0:n+1] = U[n,0:n+1].dot(1/(1+r))
        
        x=[x_0*u**i * d**(n-i) for i in range(n+1)] 
        
        U[n,0:n+1] = [max(u,gain(x[k],K)) for (k,u) in enumerate(U[n,0:n+1])]
        
    return U
    
def gain_put_vect(X,K):
    return max(np.sum(K)-np.sum(X),0)
    

def str_to_int(s):
    dico={"u":u,"d":d}

    if type(s)!=str:
        return s
    m=1
    for c in s:
        m*=dico[c]
    return m
    
start=[]
def prix_vect_am(X_0,K,N,gain):
    global start,sons_list,duplicates
    def node_payoff(X,K):
        assert len(X_0)==len(X.v)
        x=[X_0[i]*str_to_int(X.v[i]) for i in range(len(X.v))]
        return gain(x,K)
        
        
        
    init=[Node("u",p),Node("d",1-p)]
    start=init[:]
    
    if len(X_0)==1:#vectorisation si c'est un float
        start=[Node(["u"],[p]),Node(["d"],[1-p])]
        
    for i in range(1,len(X_0)):
        temp=[]     
        for j in range(len(start)):
            for k in range(len(init)):
                if type(start[j].v)==list:
                    temp+=[ Node( start[j].v+[init[k].v] , start[j].p+[init[k].p] )  ] 
                else:
                    temp+=[ Node( [start[j].v,init[k].v] , [start[j].p,init[k].p] )  ] 
        start=temp[:]
        
    start=[start[:]] 
    
    duplicates=[dict() for i in range(N+1)] # dictionnary of duplicate nodes pointing to the original (duplicate as in they have the same sons)
    sons_list=[dict() for i in range(N+1)] # index of the first son for the nodes in start

    for n in range(1,N):
        temp=[]
        #removing duplicate sons 
        
        son_id=0 #id of the first son
        
        sons_nb=len(start[0]) #number of sons
                
        for i in range(len(start[-1])):
            
            exists=False
            next_son=[start[-1][i]*node for node in start[0]]
            next_son[0].father=i #father of this branch
            
            # if next_son in temp:
            #     duplicates[n][i] = temp[temp.index(next_son)][0].father
            # else:
            #     temp.append(next_son)
            #     sons_list[n][i] = son_id
            #     son_id+=sons_nb
                
            for k in range(len(temp)):
                if next_son==temp[k]:
                    duplicates[n][i] = temp[k][0].father #pointing to the father of the identical branch<
                    exists=True
                    break
            if not exists:
                temp.append(next_son)
                sons_list[n][i]=son_id
                son_id+=sons_nb


            
        start.append(np.array(temp).flatten())
        
        
        print(n,len(start[-1]))
    sons_list[0][0] = 0
        
    start=[[Node([1 for x in X_0],[1 for x in X_0])]]+start

    U=np.zeros((N+1,len(start[-1])))
    
    U[N,:]=[node_payoff(node,K) for (i,node) in enumerate(start[-1])] #on exercice à l'étape N
    
    for n in range(N-1,-1,-1):
        
        #prix si on n'exerce pas (espérance des fils)
        print(n,len(start[n]))
        for i in range(len(start[n])):
            #moyenne des fils
            if (i in duplicates[n].keys()): #if the node is a duplicate, it gets the value of the original
                U[n,i] = U[n,duplicates[n][i]]
            else:
                for j in range(len(start[1])):
                    if sons_list[n][i]+j>=len(start[n+1]):
                        print("wtf",j,i)
                        break
                    U[n,i] += U[n+1,sons_list[n][i]+j] * start[n+1][sons_list[n][i]+j].proba()
        
        #actualisation
        U[n,0:len(start[n])] = U[n,:len(start[n])].dot(1/(1+r))
        
        
        #maximum entre l'exercice de l'action ou pas
        U[n,0:len(start[n])] = [max(U[n,i],node_payoff(node,K)) for (i,node) in enumerate(start[n])]
        
    return U

    
T = 1 #number of years
    
sigma=0.3 # Volatility 
r_0=0.03 # interest rate (per year)

N=7 #Number of steps
k=1 # Exercise time ( bermudean option )  : execution possible every k steps

r=T*r_0/N

a=sigma*sqrt(T/N) 

d=exp(a)*(1+r) # down factor
u=exp(-a)*(1+r) # up factor 

p= (1+r-d)/(u-d) 

X_0=[100,200,300] # Initial Prices
K=[100,200,300] # Strike Prices




U=prix_vect_am(X_0,K,N,gain_put_vect)

print(U[0,0])
