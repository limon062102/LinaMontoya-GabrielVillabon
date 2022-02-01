#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm 
import matplotlib.animation as anim

class Ball():
    
    # Inicializacion de la ball
    
    def __init__(self,x0,v0,a0,t,Em0,m,radius,Id):
        
        self.dt = t[1] - t[0]
        self.t = t
        
        self.x = x0
        self.v = v0
        self.a = a0
        self.Em = Em0
        
        self.xVector = np.zeros((len(t),len(x0)))
        self.vVector = np.zeros((len(t),len(v0)))
        self.aVector = np.zeros((len(t),len(a0)))
        self.Ems = np.zeros(len(t))
        self.Tstop = 100
        
        self.m = m
        self.radius = radius
        self.Id = Id
    
    # Metodos
    
    def Evolution(self,i):
        
        self.SetPosition(i,self.x)
        self.SetVelocity(i,self.v)
        self.SetEms(i,self.Em)
        
        # Euler method
        
        self.x += self.dt * self.v
        self.v += self.dt * self.a
              
    def CheckWallLimits(self,limits,e):
        
        if self.x[1] + self.radius > limits[1] and self.v[1] > 0:
            self.v[1] = self.v[1]*e
            self.Em = self.Em*e*-1
        if self.x[1] - self.radius < - limits[1] and self.v[1] < 0:
            self.v[1] = self.v[1]*e
            self.Em = self.Em*e*-1
        if self.x[0] + self.radius > limits[0] and self.v[0] > 0:
            self.v[0] = 0
            
        
    def CheckTimeStop(self,limits,i):
        
        time = 0.

        if self.x[1] - self.radius < -limits[1] and self.Em <10:
            time = self.SetTimeStop(i,self.t)
        
        return time
    
    # Setters
    
    def SetPosition(self,i,x):
        self.xVector[i] = x
    
    def SetVelocity(self,i,v):
        self.vVector[i] = v
    
    def SetTimeStop(self,i,t):
        self.Tstop = t[i]
        return self.Tstop
        
    def SetEms(self,i,Em):
        self.Ems[i] = Em
        
    # Getters
    
    def GetPositionVector(self):
        return self.xVector
    
    def GetVelocityVector(self):
        return self.vVector
    
    def GetEm(self):
        return self.Ems
    
    def GetTimeStop(self):
        return self.Tstop
    
    def GetRrVector(self):
        return self.RrVector
    
    def GetRadius(self):
        return self.radius
    
    def ReduceSize(self,factor):
        
        self.RrVector = np.array([self.xVector[0]])
        
        for i in range(1,len(self.xVector)):
            if i%factor == 0:
                self.RrVector = np.vstack([self.RrVector,self.xVector[i]])

# Discretization
dt = 0.01
tmax = 60
t = np.arange(0,tmax+dt,dt)
Limits = np.array([20.,20.])

def RunSimulationParticle(x0,v0,a0,t,Em0,m,radius,Limits,Id=1):
    
    P1 = Ball(x0,v0,a0,t,Em0,m,radius,Id)
    a = []
    
    for i in range(len(t)):
        P1.CheckWallLimits(Limits,-0.9)
        b = P1.CheckTimeStop(Limits,i)
        P1.Evolution(i)
        a.append(b)
    a = sorted(set(a))

    return P1,a

P1,a = RunSimulationParticle(np.array([-15.,5.]),np.array([1.,0.]),np.array([0.,-9.8]),t,245,1,1.,Limits,Id=1)

def ReduceTime(t,factor):
    
    P1.ReduceSize(factor)
        
    Newt = []
    
    for i in range(len(t)):
        if i%factor == 0:
            Newt.append(t[i])
            
    return np.array(Newt)

redt = ReduceTime(t,10)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)


def init():
    ax.set_xlim(-Limits[0],Limits[0])
    ax.set_ylim(-Limits[1],Limits[1])

def Update(i):
    
    plot = ax.clear()
    init()
    plot = ax.set_title(r'$t=%.2f \ seconds$' %(redt[i]), fontsize=15)
    
    x = P1.GetRrVector()[i,0]
    y = P1.GetRrVector()[i,1]
        
    vx = P1.GetVelocityVector()[i,0]
    vy = P1.GetVelocityVector()[i,1]
        
    circle = plt.Circle((x,y),P1.GetRadius(),color='k',fill=False)
    plot = ax.add_patch(circle)
        
    return plot

Animation = anim.FuncAnimation(fig,Update,frames=len(redt),init_func=init)


plt.figure()
plt.xlabel('t [s]')
plt.ylabel('Em [J]')
plt.plot(t,P1.GetEm())
plt.show

time_stop=a[1]

print('El tiempo de parada es: ' + str(time_stop))

#Los valores de la energía mecánica son positivos debido a que se manejaron como la magnitud de esta


# In[ ]:




