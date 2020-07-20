# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 09:14:10 2020

@author: barreau
"""

try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import numpy as np
import matplotlib.pyplot as plt

Nx = 150
tmax = 1

gamma = 0.01
Vf = 0.2;
L = 0.3

deltaX = 1/Nx

if Vf > 0:
    if gamma > 0:
        deltaT = 0.8*min(deltaX/Vf, deltaX**2/(2*gamma))
    else:
        deltaT = 0.8*deltaX/Vf
else:
    if gamma > 0:
        deltaT = 0.8*deltaX**2/(2*gamma)
    else:
        deltaT = 0.8*deltaX
if L != 0:
    deltaT = min(deltaT, abs(1/L))
        
Nt = int(np.ceil(tmax/deltaT))
tmax = Nt*deltaT

x = np.linspace(0, 1, Nx)
t = np.linspace(0, tmax, Nt)

# Initial condition
z0 = 1+np.sin(x*2*np.pi)
# Boundary conditions
zBC1 = 1+np.cos(2*np.pi*t)
zBC0 = (1+np.cos(2*np.pi*t))**2/2

z = np.zeros((Nx, Nt))
z[:,0] = z0
for k in range(1,Nt):
    z[0,k] = zBC0[k]
    z[-1,k] = zBC1[k]
    for i in range(1,Nx-1):
        z[i,k] = z[i,k-1] + deltaT*(gamma*(z[i-1,k-1] - 2*z[i,k-1] + z[i,k-1])/deltaX**2 \
                                    - Vf*(z[i-1,k-1] - z[i+1,k-1])/(2*deltaX) \
                                        + L*z[i,k-1])

fig = plt.figure(figsize=(7.5, 5))
X, Y = np.meshgrid(t, x)
plt.pcolor(X, Y, z)
plt.xlabel('Time')
plt.ylabel('Position')
plt.xlim(0, tmax)
plt.ylim(0, 1)
plt.colorbar()
plt.tight_layout()
plt.show()

print('max:',np.max(z))
print('min:',np.min(z))