import numpy as np
import scipy.stats as st
from multiprocessing import Pool
from functools import partial
from scipy.cluster.hierarchy import dendrogram,average,fcluster
import hcluster
from collections import Counter,defaultdict

def Clustrize(R):
    Asq = hcluster.squareform(np.sqrt(2*(1 -R)))
    S = average(Asq)
    
    d = dendrogram(S,no_plot=True)

    H = [list(fcluster(S,i,criterion='maxclust')-1) for i in range(1,R.shape[0])]+[d['leaves']]
    lvl = sorted( (1-(S[:,2]**2)/2) )+[1.]
    
    return H,lvl

def Get_Node(H):
    Nod = []
    for e in (zip(*[H[i],H[-1]]) for i in range(len(H)-1)):
        x = defaultdict(list)
        for a,b in e:
            x[a].append(b)
        x = map(tuple,x.values())
        x = filter(lambda x:len(x)>1,x)
        Nod.extend(x)
    Nod = map(set,list(set(Nod)))
    return Nod

def Find_level(H,lvl,s):
    for i in range(len(H)-1):
        h = np.array(sorted(zip(H[-1],H[i])))[:,1][s]
        t = len(set(h))
        if t==2: break
    return lvl[i-1]

def Evaluate_tree(R,X,Nt):
    H,lvl = Clustrize(R)

    C = {tuple(sorted(a)):0 for a in Get_Node(H)}

    for i in range(Nt):
        Xb = X.T[np.random.choice(range(X.shape[1]),size=X.shape[1],replace=True)].T
        Rb = np.corrcoef(Xb)

        Hb,_ = Clustrize(Rb)

        c = [tuple(sorted(a)) for a in Get_Node(Hb)]

        for cx in c:
            if C.has_key(cx):
                C[cx]+=1


    C = sorted([(1.-C[i]/float(Nt),i) for i  in C ])
    return H,lvl,C

def Obtain_FactorLoad(H,lvl,N,sel):
    S = np.zeros((N,len(sel)))

    for i in range(len(sel)):
        S[sel[i],i] = Find_level(H,lvl,sel[i])

    P = np.zeros((N,len(sel)))
    P[:,0] = np.sqrt( S[:,0] )
    for i in range(1,P.shape[1]):
        l =  S[:,i] - S[:,:i].max(axis=1) 
        l[l<0] = 0.
        l = np.sqrt(l)
        P[:,i] = l
    return P

def Create_CompleteFactors(H,lvl,N,sel):
    P = Obtain_FactorLoad(H,lvl,N,sel)

    U = np.diag(np.sqrt(1 - (P**2).sum(axis=1) ))
    B = np.hstack((P,U))
    return B

def SimulateData(B,M):
    A = st.norm.rvs(0,1,size=(B.shape[1],M))
    Xb = np.dot(B,A)
    Rb = np.corrcoef(Xb)
    return Xb,Rb


def BootFactors(B,M,Nt,sel,t):
    Xb,Rb = SimulateData(B,M)

    Hb,lvlb,Cb = Evaluate_tree(Rb,Xb,Nt)

    selb = map(np.array, sorted([b for a,b in Cb if a<=t],key=lambda x:len(x),reverse=True)) 


    selt = map(tuple,sel)
    seltb = map(tuple,selb)
    Sensitivity = sum([a in selt for a in seltb])/float(len(sel))
    Specificity = sum([a in seltb for a in selt])/float(len(selb))
    Rmet = (Sensitivity + Specificity)/2.
    return Rmet
    
def Hierarchy(X,Nt=1000,n=10,ncpu=1,nthr=6):

    R = np.corrcoef(X)
    H,lvl,C = Evaluate_tree(R,X,Nt)

    thr = list(sorted(set(zip(*C)[0])))

    sol = []

    for t in thr[:nthr]:
        sel = map(np.array, sorted([b for a,b in C if a<=t],key=lambda x:len(x),reverse=True)) 

        B = Create_CompleteFactors(H,lvl,X.shape[0],sel)

        p = Pool(ncpu)
        par = partial(BootFactors,B,X.shape[1],Nt,sel)


        Rmet  = p.map(par,[t]*n)
        p.close()

        sol.append([(1-t)*100.,np.mean(Rmet),np.std(Rmet),B])
    return sol 
