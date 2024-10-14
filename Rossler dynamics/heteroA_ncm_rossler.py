#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.io
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv
import multiprocess as mp
import os
import pandas as pd
import ast
import random

# Load the appropriate adjacency matrix and convert it to Laplacian
Aij = np.loadtxt(open("./data/NCM/G.csv", "rb"), delimiter=",").astype("float")

# Calculate the degree vector and Laplacian matrix
degree_vector = np.sum(Aij, axis=1)
Lij = np.diag(degree_vector) - Aij

# Sparsify the Laplacian matrix
sparseL = csr_matrix(Lij)

N = sparseL.shape[0]
        
##########################################
########## ESTABLISH PARAMETERS ##########
##########################################
# Load the Rossler initial condition (within the attractor)
initc = [0.1, 0.1, 0.1]

#length = 500  # Number of couplings (lambda) to calculate over 
lambdas = np.arange(0, 1.5, 0.001)
length = len(lambdas)
##########################################
########## EQUATIONS AND SOLVER ##########
##########################################

# Establish ODE 
def rossler(t, u, lambda_, a=0.1, b=0.1, c=18):
    u = u.reshape(-1, 3)  # Reshape back to (N, 3) for processing
    coupling = sparseL.dot(u[:, 1])  # Matrix-vector multiplication
    du = np.zeros_like(u)
    du[:, 0] = -u[:, 1] - u[:, 2]
    du[:, 1] = u[:, 0] + a * u[:, 1] - lambda_ * coupling
    du[:, 2] = b - c * u[:, 2] + u[:, 2] * u[:, 0]
    return du.flatten()

# Function to solve the ODE for a specific lambda
# varying parameters a,b,c
def solve_ode(lambda_, u0):
    t_span = (0, 1500)
    t_eval = np.arange(1400, 1500, 0.25)
    sol = solve_ivp(rossler, t_span, u0, vectorized=True, args=(lambda_,), t_eval=t_eval)
    return sol.y  # Return only the y coordinate

t_eval = np.arange(1400, 1500, 0.25)

##### HERE  #####
n_realizations = 500
###### HERE ######

df = pd.read_csv("./data/NCM/pred_clusters.csv")
clusters = df.cluster.tolist()
# Convert to list of lists
clusters = [ast.literal_eval(s) for s in clusters]

clusters_dict= {}
clusters_errdict = {}
count=1
for i in range(len(clusters)):
    if len(clusters[i])>1:
        #print(clusters[i])
        clusters_dict["C%d"%(count)] = clusters[i]
        clusters_errdict["C%d"%(count)] = np.zeros((n_realizations, len(lambdas)))
        count+=1

# compute global synchronization error
clusters_dict["global_error"] = list(range(1,N+1))
clusters_errdict["global_error"] = np.zeros((n_realizations, len(lambdas)))

def cluster_error(Y):
    # Y : 2 dimension matrix(nodes in cluster, time)
    # (10,3,400)
    y = np.copy(Y)
    meany = np.mean(y, axis=0)
    err_square = np.sum((y - meany)**2, axis=1)# euclidean distance squared
    #err_square = (y - meany)**2
    rms = np.sqrt(np.mean(err_square, axis=0))
    time_avg = np.mean(rms)
    return time_avg

if __name__ == "__main__":
    #print("Number of threads available:", mp.cpu_count())
    
    for nr in range(n_realizations):
        
        # Parameters and initial conditions
        x0 = initc[0] + 0.01 * np.random.rand(N)
        y0 = initc[1] + 0.01 * np.random.rand(N)
        z0 = initc[2] + 0.01 * np.random.rand(N)
        u0 = np.column_stack((x0, y0, z0))

        # Network heterogeneity, 10% randomness 
        Aij = np.loadtxt(open("./data/NCM/G.csv", "rb"), delimiter=",").astype("float")

        for i in range(Aij.shape[0]):
            for j in range(i+1, Aij.shape[1]):
                #if Aij[i,j]!=0 and bool(random.getrandbits(1)):
                if bool(random.getrandbits(1)):
                    rnd = np.random.uniform(Aij[i,j]*0.90, Aij[i,j]*1.1) 
                    Aij[i,j] = rnd
                    Aij[j,i] = rnd

        # Calculate the degree vector and Laplacian matrix
        degree_vector = np.sum(Aij, axis=1)
        Lij = np.diag(degree_vector) - Aij
        
        # Sparsify the Laplacian matrix
        sparseL = csr_matrix(Lij)

        with mp.Pool(processes=30) as pool:
            results = pool.starmap(solve_ode, [(lambdas[i], u0.flatten()) for i in range(length)])

        # Convert results to a numpy array
        sim = np.array(results)
        
        # reshaped to (200,10,3,400) and store only y coordinates
        sim = np.reshape(sim, (len(lambdas),N,3,len(t_eval)) )
        
        # compute cluster error dict
        for cl_num, cl in clusters_dict.items():
            cluster_indices = np.array(cl)-1
        
            # compute error
            for j in range(len(lambdas)):        
                Y = sim[j, cluster_indices, :, :]
                clusters_errdict[cl_num][nr,j] = cluster_error(Y)
            
        if (nr+1)%100==0:
            print(nr+1)
        #print(nr+1)
        
    np.savez("./data/NCM/heteroA_errdict3.npz", **clusters_errdict)

