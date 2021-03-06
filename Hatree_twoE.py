"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

Created on Tue Jan 30 23:38:27 2018

@author: Mark Kamper Svendsen

Modified Hatree code from the electronic structure course


# This script solves the Hartree equations for two interacting electrons in 
# a one-dimensional infinite potential well. The strength of the
# electron-electron interaction can by tuned by the parameter 'gamma'.

"""

# <codecell>

# Imports

import numpy as np
from scipy import linalg as LA
import scipy.sparse as sp
import matplotlib.pyplot as plt

# Definitions

hbar = 1   # Plancks constant
m = 1      # electron mass
L = 0.5    # Length of potential well
e = 1      # Electron charge

  
xmin=-0.1  # minimum x-coordinate
xmax=L+0.1 # maximim x-coordinate
Vinf = 1e9 # value of (external) potential outside the well

gamma = 1e10  # Scaling of electron-electron interaction [ we use 1/|x-x'| --> V_delta*delta(x-x') ]

mixer = 0.2
dx = 0.001 # discretization in x

# <codecell>
#=============================================================================#
# Defining the problem (Read potential)

x = np.arange(xmin,xmax,dx)
nx = len(x)
vext = 0*x  # External potential
n1 = np.where(x < 0) # 
n2 = np.where(x > L) #

vext[n1] = Vinf # Walls
vext[n2] = Vinf # Walls
Vext = np.diag(vext) # External potential as a matrix (operator)

# <codecell>
# Kinetic energy operator

T = np.eye(nx) 

# Redo loop for better performance

for i in range(nx-1):
    T[i,i+1]= -0.5
    T[i+1,i]= -0.5

T = T/(dx**2)*hbar**2/m

# <codecell>

H = T + Vext   # Non-interacting Hamiltonian

# <codecell>

En,Psi0 = LA.eigh(H,eigvals=(0,5))

# <codecell>
plt.figure(1)
for i in range(5):
    plt.plot(x, Psi0[:,i])
plt.title("Non-interacting wavefunctions")
plt.xlabel("Position")
plt.ylabel("Wavefunction")
plt.legend(["GS", "ES1","ES2","ES3","ES4","ES5"])


# <codecell>
# Start of Hartree part
count = 1
psi_diff = 100
tol = 1e-3  # convergence criterion for the difference in wave functions

# <codecell>
# Taking two electrons <--- one in GS and one in ES1
psi1 = Psi0[:,1]
psi2 = Psi0[:,3]
psi3 = Psi0[:,0]

# <codecell>

# Calculating psi and V_eff recursively until convergence is met. 

while (count < 100) and (psi_diff > tol):
    n1 = abs(psi1**2)
    n2 = abs(psi2**2)
    n3 = abs(psi3**2)
    
    # Setup effective potential for electron '1' and '2'
    v1 = (n2+n3)*gamma     
    V1 = np.diag(v1)
    
    v2 = (n1+n3)*gamma  
    V2 = np.diag(v2)
    
    v3 = (n1+n2)*gamma  
    V3 = np.diag(v3)
    
    # New hamiltonians for '1', '2' and '3'
    H1 = T + Vext + V1
    H2 = T + Vext + V2
    H3 = T + Vext + V3
    
    e1,Psi1 = LA.eigh(H1,eigvals=(0,5))
    e2,Psi2 = LA.eigh(H2,eigvals=(0,5))
    e3,Psi3 = LA.eigh(H3,eigvals=(0,5))
    e1 = e1[0]
    psi1_new = Psi1[:,0]
    e2 = e2[0]
    psi2_new = Psi2[:,0]
    e3 = e3[0]
    psi3_new = Psi3[:,0]
    
    
    # Computing convergence criterium
    psi_diff1 = sum(abs(abs(psi1)-abs(psi1_new)))
    psi_diff2 = sum(abs(abs(psi2)-abs(psi2_new)))
    psi_diff3 = sum(abs(abs(psi3)-abs(psi3_new)))
    psi_diff = max(psi_diff1, psi_diff2, psi_diff3)
    count = count+1    
    
    # Computing the new wavefunctions
    psi1 = psi1*(1-mixer) + psi1_new*mixer
    psi2 = psi2*(1-mixer) + psi2_new*mixer
    psi3 = psi3*(1-mixer) + psi3_new*mixer
    psi1 = psi1/LA.norm(psi1)
    psi2 = psi2/LA.norm(psi2)
    psi3 = psi3/LA.norm(psi3)

plt.figure(2)
plt.plot(x, psi1,'r',x, psi2,'g', x, psi3, 'b')
plt.title("Wavefunctions calculated using the Hatree method")
plt.xlabel("Position")
plt.ylabel("\Psi")