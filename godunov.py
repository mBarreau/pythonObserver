# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:29:12 2020

@author: barreau
"""

import numpy as np
import matplotlib.pyplot as plt

class PhysicsSim:
    
    def __init__(self, L, Nx, Tmax, Vf=1, gamma=0.05):
        self.Nx = Nx
        self.L = L
        self.Tmax = Tmax
        self.update(Vf, gamma)
        
    def update(self, Vf, gamma):
        self.Vf = Vf
        self.gamma = gamma
        self.deltaX = self.L/self.Nx
        self.deltaT = 0.8*min(self.deltaX/Vf, self.deltaX**2/(2*gamma))
        self.Nt = int(np.ceil(self.Tmax/self.deltaT))
        
class ProbeVehicles:
    
    def __init__(self, sim, xiPos, xiT):
        self.sim = sim
        self.xiPos = xiPos
        self.xiT= xiT
        self.Nxi = len(xiPos)
        self.xi = np.zeros((self.Nxi, self.sim.Nt))
        self.xi[:,0] = self.xiPos*sim.Nx/sim.L
        self.xi = np.array(self.xi, dtype=int)
        self.xiArray = np.zeros((self.Nxi, sim.Nt))
        for j in range(self.Nxi):
            self.xiArray[j, 0] = self.xiPos[j]
            
    def update(self, z, n):
        
        for j in range(self.Nxi): # ODE for the probes vehicles
            if self.xi[j,n] >= self.sim.Nx or n*self.sim.deltaT < self.xiT[j]:
                self.xiArray[j, n] = np.nan
                continue
            if (n-1)*self.sim.deltaT < self.xiT[j] and n*self.deltaT >= self.xiT[j]:
                self.xiArray[j, n-1] = self.xiPos[j]
            self.xiArray[j, n] = self.xiArray[j, n-1] + self.sim.deltaT*self.speed(z[self.xi[j,n-1]])
            self.xi[j,n] = self.xiArray[j, n]*self.sim.Nx/self.sim.L
            
    def speed(self, z):
        return self.sim.Vf*(1-z)
    
    def getMeasurements(self, z, xMeasurements = [], tMeasurements = [], zMeasurements = []):
        
        N = self.sim.Nt * self.Nxi + len(xMeasurements)
        for n in range(self.sim.Nt):
            for j in range(self.Nxi):
                if np.isnan(self.xiArray[j, n]) == False:
                    xMeasurements.append(self.xiArray[j,n])
                    tMeasurements.append(n*self.sim.deltaT)
                    zMeasurements.append(z[self.xi[j,n],n])
                    
        return (np.array(xMeasurements).reshape(N, 1), \
                np.array(tMeasurements).reshape(N, 1), \
                np.array(zMeasurements).reshape(N, 1))
    
    def plot(self, t):
        it = np.round(np.arange(0, self.sim.Nt, self.sim.Nt/len(t))).astype(int)
        xiArrayPlot = self.xiArray[:, it]
        for i in range(self.Nxi):
            plt.scatter(t, xiArrayPlot[i, :], s=1, color='red')
        

class BoundaryConditions:
    
    def __init__(self, sim, minZ0, maxZ0, sinePuls=15):
        self.minZ0 = minZ0
        self.maxZ0 = maxZ0
        self.sinePuls = sinePuls
        self.sim = sim
    
    def getZ0(self):
        Nx = self.sim.Nx
        L = self.sim.L
        averageSine = (self.maxZ0 + self.minZ0)/2
        amplitudeSine = (self.maxZ0 - self.minZ0)/2
        Nx1, Nx2, Nx3 = (int(np.floor(0.12*Nx/L)), int(np.floor(0.6*Nx/L)), int(np.floor(3*Nx/L)))
        Nx4 = Nx - Nx1 - Nx2 - Nx3
        angleSine = np.vstack(self.sinePuls*np.sqrt(np.arange(Nx3)*L/Nx))
        z0 = np.concatenate((averageSine*np.ones((Nx3, 1)) + amplitudeSine*np.cos(angleSine),
                             np.ones((Nx2, 1))*self.minZ0, (self.minZ0+2*self.maxZ0)/3*np.ones((Nx1, 1)),
                             (2*self.minZ0+self.maxZ0)/3*np.ones((Nx4, 1))), axis=0)
        z0 = 0.3*np.ones((Nx,1))
        return z0
    
    def getZin(self):
        Nt = self.sim.Nt
        Tmax = self.sim.Tmax
        angleCos = np.vstack(self.sinePuls*np.sqrt(np.arange(Nt)*Tmax/Nt))
        zin = np.ones((Nt, 1))*self.minZ0 + \
            (self.maxZ0 - self.minZ0)*(np.cos(angleCos)+1)/2
        return zin
    
    def getZout(self):
        Nt = self.sim.Nt
        Tmax = self.sim.Tmax
        angleCos = np.vstack(self.sinePuls*np.sqrt(np.arange(Nt)*Tmax/Nt))
        zout = np.ones((Nt, 1))*self.minZ0 + \
            (self.maxZ0 - self.minZ0)*(np.cos(angleCos)+1)/2
        return zout
    
class SimuGodunov:

    def __init__(self, Vf, gamma, zMin, zMax, xiPos, xiT, L=5, Tmax = 2, Nx = 300):
        
        self.sim = PhysicsSim(L, Nx, Tmax, Vf, gamma)
        
        bc = BoundaryConditions(self.sim, zMin, zMax)
        
        self.z0 = bc.getZ0()
        
        self.zin = bc.getZin()
        self.zin = 0.2*np.ones((self.zin.shape))
        self.zout = bc.getZout()
        self.zout = 0.4*np.ones((self.zin.shape))
        
        self.zMax = zMax
        self.zMin = zMin
        
        self.pv = ProbeVehicles(self.sim, xiPos, xiT)
        
    
    def simulation(self):
        
        Nx = self.sim.Nx
        Nt = self.sim.Nt
        deltaX = self.sim.deltaX
        deltaT = self.sim.deltaT
        Vf = self.sim.Vf
        gamma = self.sim.gamma
        
        z = np.zeros((Nx, Nt))
        
        for i in range(Nx):
            z[i, 0] = self.z0[i]
    
        for n in range(1, Nt): # Apply Godunov scheme
                
            z[0, n] = self.zin[n]
            
            for i in range(1, Nx-1): # Real traffic state
        
                # Heat equation
                z[i, n] = z[i, n-1] + deltaT*(gamma*(z[i-1, n-1] -
                            2*z[i, n-1] + z[i+1, n-1])/deltaX**2 -
                            Vf*(1-2*z[i, n-1])*(z[i+1, n-1] - z[i-1, n-1])/(2*deltaX))
                z[i, n] = min(z[i, n], self.zMax)
                z[i, n] = max(z[i, n], self.zMin)
            
            z[-1, n] = self.zout[n]
        
            
            self.pv.update(z[:, n], n)
        
        self.z = z
        return z
    
    def getAxisPlot(self):
        return (self.x, self.t)
    
    def plotPoints(self, x, t):
        plt.scatter(t, x, s=1, color='red')
        
    def plot(self, NxPlot=-1, NtPlot=-1, vmin=np.nan, vmax=np.nan):
        
        z = self.z
        
        if NxPlot < 0:
            NxPlot = min(4000, self.sim.Nx)
        if NtPlot < 0:
            NtPlot = min(700, self.sim.Nt)
            
        self.t = np.linspace(0, self.sim.Tmax, NtPlot)
        self.x = np.linspace(0, self.sim.L, NxPlot)
        it = np.round(np.arange(0, self.sim.Nt, self.sim.Nt/NtPlot)).astype(int)
        ix = np.round(np.arange(0, self.sim.Nx, self.sim.Nx/NxPlot)).astype(int)
        zPlot = z[ix, :]
        zPlot = zPlot[:, it]
            
        fig = plt.figure(figsize=(7.5, 5))
        X, Y = np.meshgrid(self.t, self.x)
        
        if np.isnan(vmin) or np.isnan(vmax):
            plt.pcolor(X, Y, zPlot)
        else:
            plt.pcolor(X, Y, zPlot, vmin=vmin, vmax=vmax)
        
        plt.xlabel('Time [h]')
        plt.ylabel('Position [km]')
        plt.xlim(0, self.sim.Tmax)
        plt.ylim(0, self.sim.L)
        plt.colorbar()
        plt.tight_layout()
        # fig.savefig('density.eps', bbox_inches='tight')
        plt.show()
        self.pv.plot(self.t)
        
    def getMeasurements(self, boundaryValues=False):
        
        xMeasurements = []
        tMeasurements = []
        zMeasurements = []
        if boundaryValues == True:
            for n in range(self.sim.Nt):
                xMeasurements.append(0)
                tMeasurements.append(n*self.sim.deltaT)
                zMeasurements.append(self.z[0,n])
                
                xMeasurements.append(self.sim.L)
                tMeasurements.append(n*self.sim.deltaT)
                zMeasurements.append(self.z[-1,n])
                
        measurementsFromPV = self.pv.getMeasurements(self.z, xMeasurements, tMeasurements, zMeasurements)
                
        return measurementsFromPV