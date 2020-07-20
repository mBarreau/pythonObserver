# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:23:33 2020

@author: barreau
"""

import numpy as np
import godunov as g
import neural_network as nn

# General parameters
Vf = 1 
gamma = 0.005
zMin = 0.1
zMax = 0.8

# Initial position and time of probes vehicles
xiPos = np.array([3.5, 2, 1.1, 0.5]) # Decreasing function
xiT = np.array([0, 0, 0, 0])

simu_godunov = g.SimuGodunov(Vf, gamma, zMin, zMax, xiPos, xiT)
z = simu_godunov.simulation()
simu_godunov.plot()

N_alea = 500;
x_alea = np.random.randint(0, simu_godunov.sim.Nx, N_alea)
t_alea = np.random.randint(0, simu_godunov.sim.Nt, N_alea)
z_alea = [z[x,t] for x,t in zip(x_alea, t_alea)]
neural_network = nn.NeuralNetwork([50,50,50,50,50,50,50,50])
x_alea = x_alea*simu_godunov.sim.L/simu_godunov.sim.Nx
t_alea = t_alea*simu_godunov.sim.Tmax/simu_godunov.sim.Nt
neural_network.fit(x_alea, t_alea, z_alea, epochs=200)
neural_network.plot(simu_godunov.getAxisPlot())
simu_godunov.plotPoints(x_alea, t_alea)

neural_network = nn.NeuralNetwork([50,50,50,50,50,50,50,50])
(x_train, t_train, z_train) = simu_godunov.getMeasurements()
neural_network.fit(x_train, t_train, z_train, epochs=300)
neural_network.plot(simu_godunov.getAxisPlot())
simu_godunov.pv.plot(simu_godunov.getAxisPlot()[1])

neural_network.updateMu(mu2=0.2)
neural_network.fit(x_train, t_train, z_train, epochs=300)
neural_network.plot(simu_godunov.getAxisPlot())
simu_godunov.pv.plot(simu_godunov.getAxisPlot()[1])
