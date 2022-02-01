#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm 
import matplotlib.animation as anim

class Particle():
    
    # init
    def __init__(self, x0,v0,a0,t,m,radius,Id):
        
        self.dt  = t[1] - t[0]
        
        self.x = x0
        self.v = v0
        self.a = a0
        
        self.xVector = np.zeros((len(t),len(x0)))
        self.vVector = np.zeros((len(t),len(v0)))
        self.aVector = np.zeros((len(t),len(a0)))
        
        self.m = m
        self.radius = radius
        self.Id = Id
        
    # Method
    def Evolution(self,i):
        
        self.SetPosition(i,self.x)
        self.SetVelocity(i,self.v)
        
       # print(self.r)
        
        # Euler method
        self.x += self.dt * self.v
        self.v += self.dt * self.a
    
    def CheckWallLimits(self,limits,dim=3):
        
        for i in range(dim):
            
            if self.x[i] + self.radius > limits[i]:
                self.v[i] = - self.v[i]
            if self.x[i] - self.radius < - limits[i]:
                self.v[i] = - self.v[i]
    
    # Setters
    
    def SetPosition(self,i,x):
        self.xVector[i] = x
        
    def SetVelocity(self,i,v):
        self.vVector[i] = v
        
    # Getters  
    def GetPositionVector(self):
        return self.xVector
    
    def GetRPositionVector(self):
        return self.RrVector 
    

    def GetVelocityVector(self):
        return self.vVector
    
    def GetR(self):
        return self.radius
    
    def ReduceSize(self,factor):
        
        self.RrVector = np.array([self.xVector[0]])
        
        for i in range(1,len(self.xVector)):
            if i%factor == 0:
                self.RrVector = np.vstack([self.RrVector,self.xVector[i]])

# Discretization
dt = 0.01
tmax = 10
t = np.arange(0,tmax+dt,dt)

def GetParticles(NParticles,Limit,Velo,Dim=3,dt=0.1):
    
    Particles_ = []
    
    for i in range(NParticles):
        
        x0 = np.random.uniform(-Limit+1.0,Limit-1.0,size=Dim)
        v0 = np.random.uniform(-Velo,Velo,size=Dim)
        a0 = np.zeros(Dim)
        
        p = Particle(x0,v0,a0,t,1.,1.0,i)
        
        Particles_.append(p)
        
    return Particles_

Limits = np.array([10.,10.,10.])

def RunSimulation(t,NParticles=100, Velo=6):
    
    Particles = GetParticles(NParticles,Limits[0],Velo=Velo,dt=dt)
    
    for it in tqdm(range(len(t))): # Evolucion temporal
        for i in range(len(Particles)):
            
            Particles[i].CheckWallLimits(Limits)
            Particles[i].Evolution(it)
        
    return Particles

Particles = RunSimulation(t,100,Velo=6)

def ReduceTime(t,factor):
    
    for p in Particles:
        p.ReduceSize(factor)
        
    Newt = []
    
    for i in range(len(t)):
        if i%factor == 0:
            Newt.append(t[i])
            
    return np.array(Newt)

redt = ReduceTime(t,10)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')


def init():
    ax.set_xlim(-Limits[0],Limits[0])
    ax.set_ylim(-Limits[1],Limits[1])
    ax.set_zlim(-Limits[2],Limits[2])

def Update(i):
    
    plot = ax.clear()
    init()
    plot = ax.set_title(r'$t=%.2f \ seconds$' %(redt[i]), fontsize=15)
    
    for p in Particles:
        x = p.GetRPositionVector()[i,0]
        y = p.GetRPositionVector()[i,1]
        z = p.GetRPositionVector()[i,2]
        
        vx = p.GetVelocityVector()[i,0]
        vy = p.GetVelocityVector()[i,1]
        vz = p.GetVelocityVector()[i,2]
        
        circle = plt.Circle((x,y,z),p.GetR(),color='k',fill=True)
        plot = ax.plot(x,y,z,linestyle="", marker="o")
        
        
    return plot

Animation = anim.FuncAnimation(fig,Update,frames=len(redt),init_func=init)


# In[ ]:




