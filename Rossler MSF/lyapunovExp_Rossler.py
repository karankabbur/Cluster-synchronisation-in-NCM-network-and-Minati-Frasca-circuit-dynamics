#!/usr/bin/env python
# coding: utf-8

import numpy as np
import json
import os
import multiprocess as mp
from functools import partial

if not os.path.exists("./data/"): 
    os.makedirs("./data/") 
if not os.path.exists("./data/msf/"): 
    os.makedirs("./data/msf/") 


def RK_step(f, params, ti, p, h):
    """
    Compute a single Runge-Kutta step with the function f, parameters params, 
    at time ti with values given by the vector p and a spacing h.
    """
    k1 = f(ti, p, params)
    k2 = f(ti + 0.5 * h, p + 0.5 * h * k1, params)
    k3 = f(ti + 0.5 * h, p + 0.5 * h * k2, params)
    k4 = f(ti + h, p + h * k3, params)
    
    return p + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def compute_RK4(f, params, p0, t, h, N, frec):
    """
    Compute a Runge-Kutta step with the function f, with parameters params, 
    during time t, with timestep h starting from vector p0, over N iterations, 
    and reset after frec iterations.
    """
    P = np.zeros((2, 6))  # Only store current and next step
    P[0, :] = p0          # Initial condition
    
    num_measurements = N // frec
    enorm = np.zeros(num_measurements)  # initialization of the error norm's vector
    
    # Runge-Kutta
    index = 0
    for it in range(N):
        if it > 0:
            P[0, :] = P[1, :]  # Move next step to current
        
        ti = t[it]
        P[1, :] = RK_step(f, params, ti, P[0, :], h)
        
        # Measure of the error vector norm every frec iterations
        if (it + 1) % frec == 0:
            if index >= num_measurements:
                break
            
            enorm[index] = np.log(np.linalg.norm(P[1, 0:3]))/ (frec * h)  # log of error vector norm divided by elapsed time 
            
            # Normalize the error vector for the next iteration
            P[1, 0:3] = P[1, 0:3] / np.linalg.norm(P[1, 0:3])
            
            index += 1
    
    return enorm

def LyapExp(nu, f, parameters, initc, t, h, N, frec, Tsample):
    """
    Compute the Lyapunov exponent of the dynamical system f(x) with parameters,
    couplings nulist, from an initial condition initc, during N iterations with h timestep, 
    calculated at a frequency frec, and averaged over the last Tsample values.
    """
    # initialization p0(1:3,1) random. p0(4:6,1) initc
    p0 = np.zeros(6)
    p0[3:6] = initc  # Decoupled system
    d = np.random.rand(3)
    p0[0:3] = d/np.linalg.norm(d)  # Error vector normalised

    # E.g. the a,b,c of Rossler or the sigma, rho, beta of Lorenz
    params = (parameters[0], parameters[1], parameters[2], nu)

    enorm = compute_RK4(f, params, p0, t, h, N, frec)
    explya = np.mean(enorm[-Tsample:])# averaged over M #/(tl[-1])
        
    return explya#, P, tl

# Function to compute the Rossler system
def f(t, p, params):
    dx, dy, dz, x, y, z = p
    a, b, c, nu = params

    return np.array([
        - dy - dz,
        dx + (a-nu) * dy,
        z * dx + (x - c) * dz,  # error
        - y - z,
        x + a * y,
        b + (x - c) * z  # decoupled
    ])
    
# Load the Rossler initial condition (within the attractor)
initc = [0.1, 0.1, 0.1]

# Time
t0 = 0           # initial t
T = 1000         # Tmax

h=0.001 # step size of Runge Kutta

N = int((T-t0)/h) # total number of initial iterations

frec = 50  # frequency at which we measure/calculate the Lyapunov exponent
Tsample = 100000  # Late times used for the average

# imp step to increase the number of iterations to 
# hold account of late times calculation of log error norm
N = N + Tsample*frec
T = int(N*h+t0)
t = np.linspace(t0, T, N) 

    
if __name__ == "__main__":
    print("msf2 started")
    a,b,c = 0.2, 0.2, 9
    parameters = (a,b,c)
    nulist = np.arange(0, 30, 0.1)
    
    paral_func1 = partial( LyapExp, f=f, parameters=parameters, \
                          initc=initc, t=t, h=h, N=N, frec=frec, Tsample=Tsample)
     
    with mp.Pool(processes=86) as pool:
        results = pool.map(paral_func1, nulist)
    
    msf_dict = {"parameters":parameters,
               "coupling":list(nulist),
               "msf":list(results)}
    
    with open("./data/msf/rossler_msf2.json", "w") as file:
        json.dump(msf_dict, file, indent=4)
    print("msf2 finished")

    print("msf3 started")
    a,b,c = 0.1, 0.1, 18
    parameters = (a,b,c)
    nulist = np.arange(0, 30, 0.1)
    
    paral_func1 = partial( LyapExp, f=f, parameters=parameters, \
                          initc=initc, t=t, h=h, N=N, frec=frec, Tsample=Tsample)
     
    with mp.Pool(processes=86) as pool:
        results = pool.map(paral_func1, nulist)
    
    msf_dict = {"parameters":parameters,
               "coupling":list(nulist),
               "msf":list(results)}
    
    with open("./data/msf/rossler_msf3.json", "w") as file:
        json.dump(msf_dict, file, indent=4)
    print("msf3 finished")




