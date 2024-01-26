
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:32:05 2023

@author: Nathan
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.colors as mcolors

def Load(file_name):
    path = "G:\Mon Drive\COURS\OSS\OS11\OS11 - Projet/"
    com_path = path+file_name+".txt"
    n,W = np.loadtxt(com_path,max_rows=1)
    n=int(n)
    depot = np.loadtxt(com_path,max_rows=1,skiprows=1)
    if len(depot)==2 : depot = np.append(depot,0)
    Instance = np.loadtxt(com_path,skiprows=2)
    Instance = np.insert(Instance, [0], depot, axis=0)
    
    return n,W,Instance
    
    
def print_inst(X,Y,Q):
    plt.scatter(X,Y)
    for i in range(len(X)):
        plt.text(X[i],Y[i]+0.2,str(i))
    plt.scatter(X[0],Y[0],c="red")
    plt.show()
    
def Perf(X,Y,L):
    Xa = X[L]
    #print(Xa,X)
    Ya = Y[L]
    res=0
    for i in range(len(X)):
        res+=math.sqrt((Xa[i]-Xa[i+1])**2+(Ya[i]-Ya[i+1])**2)
    return res

def PPV(X,Y,C):
    D = C.copy()      
    D[D==0]=400
    lis = [0]
    for i in range(len(X)):
        D[:,lis[i]]= 400
        min_ind = int(np.argmin(D[lis[i],:]))
        lis.append(min_ind)
    Per= Perf(X,Y,lis)
    print("Perf with PPV :",Per)
    print_res(X, Y, lis,tit = "PPV, Perf ="+str(Per))
    return lis

def KPPV(X,Y,C,k=2,n=10):
    G_perf = []
    G_lis = []
    for j in range(n):
        D = C.copy()      
        D[D==0]=800
        lis = [0]
        for i in range(len(X)):
            D[:,lis[i]]= 800
            temp = D[lis[i],:]
            min_ind=[]
            for j in range(k):
                min_ind.append( np.argmin(temp))
                temp[min_ind[j]]=400
            a = np.random.randint(0,k)
            lis.append(min_ind[a])
        Perfo = Perf(X,Y,lis)
        G_perf.append(Perfo)
        G_lis.append(lis)
    s = np.argmin(G_perf)
    print_res(X, Y, G_lis[s],tit = "KPPV,k="+str(k)+" , Perf ="+str(G_perf[s]))
    return G_lis[s]

def KPPV_one(X,Y,C,k=2):
    D = C.copy()      
    D[D==0]=800
    lis = [0]
    for i in range(len(X)):
        D[:,lis[i]]= 800
        temp = D[lis[i],:]
        min_ind=[]
        for j in range(k):
            min_ind.append( np.argmin(temp))
            temp[min_ind[j]]=400
        a = np.random.randint(0,k)
        lis.append(min_ind[a])
    Perfo = Perf(X,Y,lis)
    #print_res(X, Y, G_lis[s],tit = "KPPV,k="+str(k)+" , Perf ="+str(G_perf[s]))
    return lis


def Dist(X,Y):
    C = np.zeros((len(X),len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            C[i][j]=math.sqrt((X[i]-X[j])**2+(Y[i]-Y[j])**2)
    return C

def print_res(X,Y,L,tit=""):
    Xa = X[L]
    Ya = Y[L]
    plt.plot(Xa,Ya)
    plt.scatter(X,Y)
    plt.scatter(X[0],Y[0],c="red")
    for i in range(len(X)):
        plt.text(X[i],Y[i]+0.2,str(i))
    plt.title(tit)
    plt.show()

def Back(S,P,n,Q):
    t = n
    T=[]
    Label = np.full((2,n+1),0)
    
    Q_t = []
    for i in range(n+1):
        Label[0][i]=i
    i=1
    while(t>0):
        temp = P[t]
        A = np.array(S[temp+1:t+1])
        A = np.insert(A,[0],0)
        A = np.append(A,0)
        T.append(A)
        Label[1][temp+1:t+1]=i
        i=i+1
        t = temp
        Q_t.append(sum(Q[A]))
    return T,Q_t,Label

def print_Whole_SPLIT(T,X,Y):
    colors = mcolors.TABLEAU_COLORS
    col_ind = list(colors)
    plt.scatter(X,Y)
    for i in range(len(X)):
        plt.text(X[i],Y[i]+0.2,str(i))
    for i in range(len(T)):
        lab = str(i)
        col = colors[col_ind[i%len(col_ind)]]
        plt.plot(X[T[i][1:-1]],Y[T[i][1:-1]],label = lab,c=col)
        plt.plot(X[T[i][0:2]],Y[T[i][0:2]],linestyle=":",c=col)
        plt.plot(X[T[i][-2:]],Y[T[i][-2:]],linestyle=":",c=col)
    plt.xlim(right=max(X)+10)
    plt.title("SPLIT Algorithm")
    plt.legend(loc=7,ncols=2,title="Tournées")
    plt.show()
    
def print_Parse_SPLIT(T,X,Y,Qt,Q,La):
    colors = mcolors.TABLEAU_COLORS
    col_ind = list(colors)
    
    #plt.scatter(X,Y,c=La[1])
    for i in range(len(T)):
        col = colors[col_ind[i%len(col_ind)]]
        plt.scatter(X,Y,c="grey")
        plt.scatter(X[T[i]].copy(),Y[T[i]].copy(),c=col)
        lab = str(i)
        plt.text(X[0],Y[0]+0.2,"O")
        for k in range (len(T[i])-2):
            plt.text(X[T[i][k+1]],Y[T[i][k+1]]+0.2,str(T[i][k+1])+str(Q[T[i][k+1]]))
        plt.plot(X[T[i][1:-1]],Y[T[i][1:-1]],label = lab,c=col)
        plt.plot(X[T[i][0:2]],Y[T[i][0:2]],linestyle=":",c=col)
        plt.plot(X[T[i][-2:]],Y[T[i][-2:]],linestyle=":",c=col)
        plt.xlim(left = 0,right=max(X)+25)
        plt.ylim(bottom = 0,top=75)
        plt.title("SPLIT Algorithm, T = "+str(i)+",Q = "+str(Qt[i]))
        plt.legend(loc=7)
        plt.show()

def SPLIT(Inst,Global_C,pri = True):
    X,Y,Q = np.hsplit(Inst,3)

    #Best tour
    S_histo = []
    P_histo = []
    V_histo = []
    for k in range(10):
        #Initiate V & P
        V = np.full(n+1,10000)
        V[0]=0
        P = np.full(n+1,-1)
        #KPPV
        S = KPPV_one(X,Y,Global_C,k=4)[:n+1]
        #Sort by S
        C = Global_C.copy()
        C = C[S]
        Q_ol=Q.copy()
        Q = Q[S]
        #Distance_mat
        Dist_bt = np.insert(C.diagonal(offset=1), [0],0)
        Dist_0= C[0,:]
        C = np.array((Dist_0,Dist_bt)).transpose()
        for i in range(1,n+1):
            j=i
            while (j<=n and sum(Q[i:j+1]) <=W):            
                if j==i:
                    poids = Q[i].copy()
                    cout = 2*C[i][0]
                else : 
                    poids += Q[j].copy()
                    cout = cout - C[j-1][0] + C[j][1] + C[j][0]
                if V[i-1]+cout < V[j] :
                    V[j]=V[i-1]+cout
                    P[j]=i-1
                j += 1
        T, Q_t,Label = Back(S,P,n,Q_ol)
        print_Whole_SPLIT(T, X, Y)        
        S_histo.append(S)
        P_histo.append(P)
        V_histo.append(V[-1])
    print(V_histo,min(V_histo))
    
    T, Q_t,Label = Back(S,P,n,Q_ol)
    
    if pri : 
        print_Whole_SPLIT(T, X, Y)
        print_Parse_SPLIT(T, X, Y,Q_t,Q_ol,La = Label)
    return P,S,T,Q_t,V

n,W,T=Load("vrp-ex7")
X,Y,Q = np.hsplit(T,3)
print_inst(X,Y,Q)
C = Dist(X,Y)
P,S,T,Qt,V = SPLIT(T,C,pri=False)