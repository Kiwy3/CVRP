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
* $q_i$ : weight of the point
### Variable
$x_{ijk}$ is the binary variable, with a value of 1 if there is an vehicule driving from i to j in vehicule k, 0 else

### Objective
The objective is to minimize $\sum_i \sum_{j \neq i} \sum_k C_{ijk}.x{ijk}$
### Constraints
* $\forall i \neq 0 : \sum_{j \neq i} \sum_k x_{ijk} = 1$ ; each client i must be visited
* $\forall k : \sum_{i \neq 0} \sum_{j \neq i} q_i x_{ijk} \leq W$ : the capacity of each trip (vehicle) k is respected
* $\forall i \neq 0, \forall k : \sum_{j \neq i} x_{ijk} = \sum_{j \neq i} x_{jik}$ : continuous trip (if vehicle k arrives at client i, it leaves it)
* $\forall S \subseteq \{ 1, 2, ...,n\}, S \neq \emptyset , \forall k : \sum_{ i \in S} \sum_{ j \in S} x_{ijk} \leq |S| - 1 $ :  no subtours in each trip

This mathematic model can be used to have optimal solution, based on exact method. But with big instance, like the ones we have, the calculation time will be too much to use it. It's why we use the SPLIT heuristics. 

## Heuristic Definition
The Split heuristic is a Route First, Cluster second heuristic. It mean that we first generate a big tour, without capacities like a TSP problem, then we cluster it into the best subtour we can. 

In our case, the Big Tour is done with the Nearest Neighbor heuristic and the Randomised Nearest Neighbors. When we have the big tour, the SPlit is done with this Pseudo-code :

>Initialiser les 𝑉𝑖 à plus l'infini, sauf 𝑉0 = 0 (labels provisoires) <br>
>Pour 𝑖 variant de 1 à 𝑛 <br>
>>  Pour 𝑗 variant de 𝑖 à 𝑛 tant que la demande de (𝑆𝑖 , … , 𝑆𝑗) est ≤ 𝑊 <br>

 >>Si 𝑖 = 𝑗 _// Tournée à un client, simple aller-retour_ <br>
 >> - 𝑝𝑜𝑖𝑑𝑠 = 𝑞𝑖 et 𝑐𝑜𝑢𝑡 = 2 × 𝐶(0, 𝑆𝑖) <br>
 >>
 >>Sinon // On ajoute le client 𝑗 à la fin de la tournée précédente <br>
 >> - 𝑝𝑜𝑖𝑑𝑠 = 𝑝𝑜𝑖𝑑𝑠 + 𝑞𝑗 <br>
 >> - 𝑐𝑜𝑢𝑡 = 𝑐𝑜𝑢𝑡 − 𝐶(𝑆𝑗−1, 0) + 𝐶(𝑆𝑗−1, 𝑆𝑗) + 𝐶(𝑆𝑗, 0) <br>
 >>Fin si <br>
 
 >> Si 𝑉𝑖−1 + 𝑐𝑜𝑢𝑡 < 𝑉𝑗 <br>
 >> - 𝑉𝑗 = 𝑉𝑖 + 𝑐𝑜𝑢𝑡 <br>
 >> - 𝑃𝑗 = 𝑖 − 1 <br>
 >> Fin si <br>
 
 >>Fin pour <br>
>Fin pour <br>

This pseudo can be used with these notations : 
* S is the big tour, Si is the index of the client in the position i in the tour
* Vi is the lowest cost possible to go to the client i
* Pi is the predecessor of the client i (used to generate the subtour)

The code can be located at XXXXXXXXXXXXXXXX. 

# Results 






