# SPLIT Heuristic

Capacitated Vehicules Routing Problems (CVRP) are common problems in Vehicules Routing Problem (VRP). In our case, we only have one depot with many points.

The instances used for test are Christofides instances, with 50 and 75 points.
## CVRP Model
To define the problem, we can use the following mathematical model : 
### Notations
* $n$ : number of clients, without the depot
* $W$ : capacity of one vehicule
* $X_i , Y_i$ : coordinates of client i
* $C_{ij}$ : distance between point i and point j (client or depot)
* $Q_i$ : weight of the point
### Variable
$x_{ijk}$ is the binary variable, with a value of 1 if there is an vehicule driving from i to j in vehicule k, the objectif is to minimize $\sum_i \sum_j \sum_k C_{ijk}.x{ijk}$
### Constraints
* $\sum_i x_{ijk} = \sum_i x_{jik} \forall j \in {1,..,n},\forall k \in {1,..,p}$ : Vehicle leaves node that it enters
* $\sum_k \sum_i x_{ijk} = 1 \forall j \in {2,..,n}$ : Ensure that every node is entered once





