# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:23:33 2020

@author: barreau
"""

import numpy as np
np.random.seed(1234)
import godunov as g
import neural_network as nn
import sys

# General parameters
Vf = 1 
gamma = 0.005
zMin = 0.2
zMax = 0.4

# Initial position and time of probes vehicles
xiPos = np.array([3.5, 2, 1.1, 0.5]) # Decreasing function
xiT = np.array([0, 0, 0, 0])

simu_godunov = g.SimuGodunov(Vf, gamma, zMin, zMax, xiPos, xiT)
z = simu_godunov.simulation()
simu_godunov.plot(vmin=zMin, vmax=zMax)

N_alea = 500;
x_alea = np.random.randint(0, simu_godunov.sim.Nx, N_alea).reshape(N_alea,1)
t_alea = np.random.randint(0, simu_godunov.sim.Nt, N_alea).reshape(N_alea,1)
z_alea = [z[x,t] for x,t in zip(x_alea, t_alea)]
z_alea = np.array(z_alea).reshape(N_alea,1)
neural_network = nn.NeuralNetwork([50,50,50,50,50,50,50,50])
x_alea = x_alea*simu_godunov.sim.L/simu_godunov.sim.Nx
t_alea = t_alea*simu_godunov.sim.Tmax/simu_godunov.sim.Nt
neural_network.fit(x_alea, t_alea, z_alea, epochs=200)
neural_network.plot(simu_godunov.getAxisPlot(), zMin, zMax)
simu_godunov.plotPoints(x_alea, t_alea)

neural_network = nn.NeuralNetwork([50,50,50,50,50,50,50,50])
(x_train, t_train, z_train) = simu_godunov.getMeasurements(boundaryValues=True)
neural_network.fit(x_train, t_train, z_train, epochs=200)
neural_network.plot(simu_godunov.getAxisPlot(), zMin, zMax)
simu_godunov.pv.plot(simu_godunov.getAxisPlot()[1])

neural_network.updateMu(mu2=0.2)
x_train2 = np.vstack((x_train, x_alea))
t_train2 = np.vstack((t_train, t_alea))
z_train2 = np.vstack((z_train, np.zeros(t_alea.shape)))
isPhysics = np.vstack((np.zeros(t_train.shape), np.ones(t_alea.shape)))
neural_network.fit(x_train2, t_train2, z_train2, isPhysics, epochs=1000)
neural_network.plot(simu_godunov.getAxisPlot(), zMin, zMax)
simu_godunov.pv.plot(simu_godunov.getAxisPlot()[1])