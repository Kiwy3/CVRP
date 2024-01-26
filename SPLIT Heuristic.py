
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:32:05 2023

@author: Nathan
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.colors as mcolors

def Load(file_name):  # Load the instance from txt file
    """Load """
    path = "instances/"
    com_path = path+file_name+".txt"
    """Extract n & W"""
    n, W = np.loadtxt(com_path, max_rows=1)
    n = int(n)
    """Extract first line"""
    depot = np.loadtxt(com_path, max_rows=1, skiprows=1)
    if len(depot) == 2:
        depot = np.append(depot, 0)
    """Load the whole instance"""
    Instance = np.loadtxt(com_path, skiprows=2)
    Instance = np.insert(Instance, [0], depot, axis=0)
    return n, W, Instance


def Inst_plot(X, Y, titre=""):  # To plot the instance
    plt.scatter(X, Y)
    for i in range(len(X)):
        plt.text(X[i], Y[i]+0.2, str(i))
    plt.scatter(X[0], Y[0], c="red")  # First point is red
    plt.title(titre)
    plt.show()


def TSP_plot(X, Y, Tour, tit=""):  # To plot TSP result
    """Sort by TSP Tour"""
    Xa = X[Tour]
    Ya = Y[Tour]
    """Plot the Tour"""
    plt.plot(Xa, Ya)
    """Plot points with index"""
    plt.scatter(X, Y)
    plt.scatter(X[0], Y[0], c="red")
    for i in range(len(X)):
        plt.text(X[i], Y[i]+0.2, str(i))
    plt.title(tit)
    plt.show()


def TSP_res(X, Y, Tour):#To Calculate distance for a TSP
    """Sort by TSP Tour"""
    Xa = X[Tour]
    Ya = Y[Tour]
    distance = 0.
    """Add the distance between i and i+1 to the result"""
    for i in range(len(X)):
        distance += math.sqrt((Xa[i]-Xa[i+1])**2+(Ya[i]-Ya[i+1])**2)
    return distance


def PPV(X, Y, distances):  # PPV Heuristic to obtain result for TSP
    """Initiate the PPV and the first point"""
    D = distances.copy()
    D[D == 0] = (max(X)+max(Y))*100
    lis = [int(0)]  # List of points
    """PPV Heuristic"""
    for i in range(len(X)):
        D[:, lis[i]] = (max(X)+max(Y))*90
        min_ind = int(np.argmin(D[lis[i], :]))
        lis.append(min_ind)
    return lis




def PPVR(X, Y, C, k=2):  # PPVR Heuristic to obtain result for TSP
    """Initiate the PPVR and the diagonal"""
    D = C.copy()
    D[D == 0] = float("inf")
    lis = [0]  # List of points
    """PPVR heuristic"""
    for i in range(len(X)-1):
        D[:, lis[i]] = float("inf")
        temp = D[lis[i], :]
        sort = np.argsort(temp)
        if len(X)-i<=k:
            k=k-1
        """Randomly choose a points"""
        random_ind = np.random.randint(0, k)
        lis.append(sort[random_ind])
    lis.append(0)
    return lis


def DISTANCES(X, Y):  # Calculate distance between each points
    C = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            C[i][j] = math.sqrt((X[i]-X[j])**2+(Y[i]-Y[j])**2)
    return C


def Back(S, P, n, Quant):  # Obtain sub-tour from Predecessor vector of SPLIT
    """Initiate"""
    t = n
    T = []
    Q_subtour = []
    """The label is the sub-tour number"""
    Label = np.full((2, n+1), 0)
    for i in range(n+1):
        Label[0][i] = i
    """While to iterate from last point to first"""
    i = 1
    while(t > 0):
        """Slice S from P[T] to t"""
        temp = P[t]  # To look at t predecessor
        A = np.array(S[temp+1:t+1])
        """Insert 0 at first and last"""
        A = np.insert(A, [0], 0)
        A = np.append(A, 0)
        T.append(A)
        """Label and subtour quantity"""
        Label[1][temp+1:t+1] = i
        Q_subtour.append(sum(Quant[A]))
        """Upgrade iteration variable"""
        i = i+1
        t = temp
    Label = Label[:,S]
    
    return T, Q_subtour, Label


def Plot_CVRP_Whole(T, X, Y, distance): #Plot result for CVRP
    """Color"""
    colors = mcolors.TABLEAU_COLORS
    col_ind = list(colors)
    """Plot point with index"""
    plt.scatter(X, Y)
    for i in range(len(X)):
        plt.text(X[i], Y[i]+0.2, str(i))
    """Plot sub tour"""
    for i in range(len(T)):
        lab = str(i)
        col = colors[col_ind[i % len(col_ind)]]
        plt.plot(X[T[i][1:-1]], Y[T[i][1:-1]], label=lab, c=col)
        plt.plot(X[T[i][0:2]], Y[T[i][0:2]], linestyle=":", c=col)
        plt.plot(X[T[i][-2:]], Y[T[i][-2:]], linestyle=":", c=col)
    plt.xlim(right=max(X)*1.4)  # To see the legend
    titre = "CVRP Result, distance = {distance:.2f}"
    plt.title(titre.format(distance=distance))
    plt.legend(loc=7, ncols=2, title="Tournées")
    plt.show()
    
def Dist_tour(X,Y,L):#To calculate distance for sub-tour
    Xa = X[L]
    Ya = Y[L]
    res=0
    for i in range(len(Xa)-1):
        res+=math.sqrt((Xa[i]-Xa[i+1])**2+(Ya[i]-Ya[i+1])**2)
    return res

def Plot_CVRP_Sub(T, X, Y, Qt, Q, La): #Plot sub-tour for CVRP
    """Color"""
    colors = mcolors.TABLEAU_COLORS
    col_ind = list(colors)
    """Loop on sub-tour"""
    for i in range(len(T)):
        col = colors[col_ind[i % len(col_ind)]]
        """Scatter point, with color for point on sub-tour"""
        plt.scatter(X, Y, c="grey")
        plt.scatter(X[T[i]], Y[T[i]], c=col)
        lab = str(i)
        plt.text(X[0], Y[0]+0.2, "O")
        """Add index & Q for each point"""
        for k in range(len(T[i])-2):
            plt.text(X[T[i][k+1]], min(Y[T[i][k+1]]+0.2,max(Y)-1),
                     str(T[i][k+1])+" - "+str(int(Q[T[i][k+1]])))
        """Plot sub-tour"""
        plt.plot(X[T[i][1:-1]], Y[T[i][1:-1]], label=lab, c=col)
        plt.plot(X[T[i][0:2]], Y[T[i][0:2]], linestyle=":", c=col)
        plt.plot(X[T[i][-2:]], Y[T[i][-2:]], linestyle=":", c=col)
        """Title, lim & legend"""
        plt.xlim(right=max(X)*1.4)
        plt.title("SPLIT Algorithm, T = "+str(i)+",Q = "+str(int(Qt[i])))
        plt.legend(loc=7)
        plt.show()


def SPLIT(X,Y,Q,iterations = 50):
    Distance_mat = DISTANCES(X, Y)
    """Initiate historic variable"""
    TSP_histo = []
    Tour_histo = []
    Pred_histo = []
    Valeur_histo = []
    Label_histo = []
    Quant_tour_histo = []
    """GLobal iteration """
    stp=0
    for k in range(iterations):
        S_new = True #To check if it's a new tour
        """Initiate V & P"""
        V = np.full(n+1, 10000.)
        V[0] = 0
        P = np.full(n+1, -1)
        """Generate a big tour"""
        if (k>iterations*0.8 and iterations>50): stp = 1
        if k > 0:
            S = PPVR(X, Y, Distance_mat, k=2+stp)[:n+1]
        else:
            S = PPV(X, Y, Distance_mat)[:n+1]
        S = np.array(S)
        """Check if it's a new tour"""
        for i in range(len(TSP_histo)):
            if (S==TSP_histo[i]).all():
                S_new = False
        if S_new: 
            """Save new tour"""
            TSP_histo.append(S)
            """Sort distance & Q by S"""
            C = Distance_mat.copy()
            C = C[S][:, S]
            Q_sorted = Q[S].copy()
            """Distance_mat"""
            Dist_bt = np.insert(C.diagonal(offset=1), [0], 0)
            Dist_0 = C[0, :]
            C = np.array((Dist_0, Dist_bt, S)).transpose()
            """SPLIT Heuristic"""
            for i in range(1, n+1):
                j = i
                while (j <= n and sum(Q_sorted[i:j+1]) <= W):
                    if j == i:
                        poids = Q_sorted[i].copy()
                        cout = 2*C[i][0]
                    else:
                        poids += Q_sorted[j].copy()
                        cout = cout - C[j-1][0] + C[j][1] + C[j][0]
                    if V[i-1]+cout < V[j]:
                        V[j] = V[i-1]+cout
                        P[j] = i-1
                    j += 1
            """Obtain Tour from SPLIT"""
            Tour, Q_tour, Label = Back(S, P, n, Q)
            """Stock key value"""
            Tour_histo.append(Tour)
            Pred_histo.append(P)
            Valeur_histo.append(V[-1])
            Label_histo.append(Label)
            Quant_tour_histo.append(Q_tour)
        if iterations>100 : 
            if ((k+1)%int(iterations/10))==0:print("Avancée : ",k+1,"/",iterations,"- résultat : ",min(Valeur_histo))
    """Save values for the best results"""
    best_ind = np.argmin(Valeur_histo)
    best_tour = Tour_histo[best_ind]
    best_val = Valeur_histo[best_ind]
    best_label = Label_histo[best_ind]
    best_Q = Quant_tour_histo[best_ind]
    return best_tour,best_val,best_label,best_Q,len(Tour_histo),Distance_mat


# n,W,data=Load("vrp-50-clients")
files_dic = {"50-clients":"vrp-50-clients","75-clients":"vrp-75-clients","ex7":"vrp-ex7"}
name = "50-clients"
n, W, data = Load(files_dic[name])
X, Y, Q = np.hsplit(data, 3)
Inst_plot(X, Y, name)
nb_it = 1000

Tour,VAL,Label,Q_sub,nb_tour,Distance = SPLIT(X,Y,Q,nb_it)
"""
its = [1,5,10,20,30,40,50,75,100,150,200,300,400,500]
VAL_its = []
for nb_it in its : 
    Tour,VAL,Label,Q_sub,nb_tour,Distance = SPLIT(X,Y,Q,nb_it)
    VAL_its.append(VAL)

plt.plot(its,VAL_its,'bo-')
plt.title("Evolution de la longueur optimale de SPLIT en fonction du nombre d'itérations")
plt.show()"""

print("----------------------------------")
print("Results for ",name," with ",nb_it," iterations")
print("Capacité max : ",int(W))
print("Nombre de tours générés : ",int(nb_tour))
print(("Distance totale : {val:.2f} ").format(val=VAL))
print("Nombre de tournées :",len(Tour))
for i in range(len(Tour)):
    print("------------ Tournées ",i," ------------")
    print("   Tour : ",Tour[i])
    print("   Poids : ",int(Q_sub[i]))
    print("   Distance :",Dist_tour(X,Y,Tour[i]))
   
Plot_CVRP_Whole(Tour, X, Y, VAL)
Plot_CVRP_Sub(Tour, X, Y,Q_sub, Q, Label)