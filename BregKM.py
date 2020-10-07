import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
import random
from scipy.spatial import distance
import copy
import time
from scipy.stats import entropy
import scipy
from sklearn import datasets
from sklearn.datasets import load_digits 

def kl_divergence(p, q):# Kullback_Leibler
	return sum(p * np.log2(1.0*p/q))
def Itaku(p, q):# Itakura-Saito
	return sum((1.0*p/q) - np.log2(1.0*p/q))
	
def Dist(X,Y,A,l):# Matrix of dissimilarities
    m=len(X);n=len(Y)
    S=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if l==0:
                S[i][j]=scipy.spatial.distance.sqeuclidean(X[i],Y[j]) 
            elif l==1:
                S[i][j]=scipy.spatial.distance.mahalanobis(X[i],Y[j],A)**2 
            elif l==2:
                S[i][j]=kl_divergence(X[i],Y[j])
            elif l==3:
                S[i][j]=Itaku(X[i],Y[j])                  
    return S 
    	

def Medoid(X,cluster,A,l):# Medoid computation for BK
    x=np.mean(X[cluster], axis=0)
    v=Dist(X[cluster],np.array([x]),A,l)
    medoid=cluster[v.argmin()]    
    return medoid  


def FastDescent(X,curr_medoids,A,l,k, maxiter=1000):# FastDescent algorithm    
    old_medoids = np.array([-1]*k) 
    new_medoids = np.array([-1]*k) 
    it=0;
    while not ((old_medoids == curr_medoids).all()) and (it<maxiter):
        ############ Assignment ############
        D=Dist(X,X[curr_medoids],A,l); it+=1
        clusters = curr_medoids[np.argmin(D, axis=1)]
         ###################################
         ############ Update ###############
        for curr_medoid in curr_medoids:
            cluster = np.where(clusters == curr_medoid)[0]
            new_medoids[curr_medoids == curr_medoid] = Medoid(X,cluster,A,l)
        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]
        ####################################
    err=np.sum(np.min(D,axis=1))  
    return curr_medoids, it,err,clusters

def BregmanKM(X,curr_medoids,l,k, maxiter=1000):# BK algorithm
    print('Method','it','Error','FD_it')
    STOPc=False;erra=np.inf;IT=0;                                                                                 
    while STOPc==False:
        M,it,err,clusters=FastDescent(X,curr_medoids,A,l,k)      
        if erra<=err:
            STOPc=True
        else:
            print("BK",IT,err,it)
            erra=err
            #################### SmartSeeding ##########################
            CL,kaddc,ClusLength,i,WCE=Split(X,M,clusters,k,A,l)                                            
            curr_medoids,je,le=Merge(X,kaddc,ClusLength,CL,k,i,A,l,WCE)
            ############################################################ 
            B=Dist(X[curr_medoids],X,A,l);erri=np.sum(np.min(B,axis=0))         
        IT+=1
    return 

def Split(X,M,clusters,k,A,lll): # Split step of SmartSeeding
    Desc=np.zeros(k);Ind1=np.zeros(k);Ind2=np.zeros(k);
    ClusLength=np.zeros(k);SiN=np.zeros(k);Si=np.zeros(k);i=0
    CL1=[[] for l in range(k)]
    CL2=[[] for l in range(k)]
    ECL=[np.zeros(2) for l in range(k)]  
    CL=[[] for l in range(k+1)]
    WCE=np.zeros(k+1)
   
    for curr_medoid in M:
        cluster = np.where(clusters == curr_medoid)[0]
        Xs=X[cluster];
        Dt=Dist(Xs,np.array([X[curr_medoid]]),A,lll)
        E2 = sum(Dt);WCE[i]=E2
        if len(cluster)>=2:
            CL[i]=cluster; ClusLength[i]=len(cluster)
            ############## D-sampling initialization #################
            M1=np.random.randint(len(cluster), size=1);
            B=Dist(Xs,np.array([Xs[M1]]),A,lll);p=np.min(B,axis=1)
            ind=np.random.choice(len(cluster), 1, p=p/sum(p))[0]
            M1=np.append(M1,ind)
           ###########################################################
            Ms,it,E1,clust = FastDescent(Xs,M1,A,lll,2)
            Desc[i]=E2-E1        
            CL2[i]=cluster[np.where(clust == Ms[1])[0]] 
            CL1[i]=cluster[np.where(clust == Ms[0])[0]]
            Si[i]=len(CL2[i]);SiN[i]=len(CL1[i])
            Ind1[i]=cluster[Ms[0]];Ind2[i]=cluster[Ms[1]]
            ECL[i][0]=sum(Dist(Xs[np.where(clust == Ms[0])[0]],np.array([Xs[Ms[0]]]),A,lll));
            ECL[i][1]=sum(Dist(Xs[np.where(clust == Ms[1])[0]],np.array([Xs[Ms[1]]]),A,lll));
        else:
            Desc[i]=0
            CL[i]=cluster
            ClusLength[i]=len(cluster)            
        i+=1
    i=np.argmax(Desc)    
    CL[i]=CL1[i];M[i]=Ind1[i]
    CL[k]=CL2[i]
    WCE[i]=ECL[i][0];WCE[k]=ECL[i][1]       
    kaddc=np.append(M,Ind2[i])
    ClusLength=np.append(ClusLength,Si[i]);
    ClusLength[i]=SiN[i]
    return CL,kaddc,ClusLength,i,WCE

    
def Merge(X,kaddc,ClusLength,CL,k,i,A,l,WCE):# Merge step of SmartSeeding
    inc=np.inf
    C=[np.mean(X[CL[j]],axis=0) for j in range(k+1)]
    S=Dist(C,X[kaddc.astype(int)],A,l);
    for j in range(k):
        for ll in range(j+1,k+1):
            if (j==i)*(ll==k)==0:
                incc=min(len(CL[j])*(S[j][ll]-S[j][j]),len(CL[ll])*(S[ll][j]-S[ll][ll]))
                cluster=np.append(CL[j],CL[ll])
                mdnjl= Medoid(X,cluster,A,l)
                if incc<inc:
                    inc=incc; je=j; le=ll
                    mdn=mdnjl
    kaddc[je]=mdn  
    kaddc=np.delete(kaddc,le)
    curr_medoids=kaddc.astype(int) 
    return curr_medoids,je,le
         
####################################################       
################## Example #########################
# l sets the metric to be used as follows:
# l=0 -> Squared Euclidean
# l=1 -> Mahalanobis
# l=2 -> Kullback-Leibler
# l=3 -> Itakura-Saito
l=0;
#################################################### 
# Dataset from sklearn.datasets library:
iris = datasets.load_digits()
D=iris.data; X=np.unique(D,axis=0)
# Number of clusters:         
K=25
# Initial set of medoids (selected uniformly ar random)
curr_medoids=np.random.choice(len(X), K, replace=False)
# If metric= Mehalanobis, compute inverse of covariance matrix
if l==1:
    S=np.cov(X.T);A=np.linalg.inv(S)
else:
    A=0
#################################################### 
BregmanKM(X,curr_medoids,l,K)