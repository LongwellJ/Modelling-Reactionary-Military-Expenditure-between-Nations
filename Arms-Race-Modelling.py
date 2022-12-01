#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as pyplot 
import numpy as np 
import numpy.random as random 
import scipy.optimize as optimize 
import scipy.integrate as integrate
from sympy.solvers import solve
from sympy import symbols


# In[ ]:


import pandas as pd
df = pd.read_csv("\C:\Users\longw\OneDrive\Desktop\ProjectData.txt")


# In[ ]:


#declaring country variables
data = np.loadtxt("C:/Users/longw/OneDrive/Desktop/Project/Data.txt")
Albania = []
Belgium = []
Bulgaria =[]
Canada =[]
Croatia =[]
Czech_Republic=[]
Denmark=[]
Estonia=[]
France=[]
Germany=[]
Greece=[]
Hungary=[]
Iceland=[]
Italy=[]
Latvia=[]
Lithuania=[]
Luxembourg=[]
Montenegro=[]
Netherlands=[]
Norway=[]
Poland=[]
Portugal=[]
Romania=[]
Russia=[]
Slovakia=[]
Slovenia=[]
Spain=[]
Turkey=[]
United_Kingdom=[]
United_States=[]
West_Germany = []

l = [Albania,Belgium,Bulgaria,Canada,Croatia,Czech_Republic,Denmark,Estonia,France,Germany,Greece,Hungary,Iceland,Italy,Latvia,Lithuania,Luxembourg,Montenegro,Netherlands,Norway,Poland,Portugal,Romania,Russia,Slovakia,Slovenia,Spain,Turkey,United_Kingdom,United_States,West_Germany]


# In[ ]:


#making list of years as well as countries join dates to Nato
ty = [x for x in range(1914,2020)]
natojoin = [2009,1949,2004,1949,2009,1999,1949,2004,1949,1955,1952,1999,1949,1949,2004,2004,1949,2017,1949,1949,1999,1949,2004,2019,2004,2004,1982,1952,1949,1949,1955]


# In[ ]:


#creating a list of each countries expenditures while they were in Nato
print(1e7)
for i in range(0,len(l)):
    for j in range(0,len(data)):
        if ty.index(natojoin[i]) <= j:
            l[i].append(data[j][i])
        else:
            l[i].append(0)
#Adding up each countries expenditure to get Nato expenditure in billions of the year 2000 US dollars
nato = []
for j in range(0,len(data)):
    ysum = 0
    for i in range(0,len(l)):
        ysum +=l[i][j]
    nato.append(ysum/(1e7))
#adding up US and Soviet data in billions of the year 2000 US dollars
soviets = []
murica = []
for i in range(0,len(data)):
    soviets.append(data[i][23]/(1e7))
    murica.append(data[i][29]/(1e7))
    
#plotting the data
pyplot.scatter(ty[38:75],nato[38:75],color='b')
pyplot.scatter(ty[38:75],soviets[38:75],color='g')
pyplot.scatter(ty[38:75],murica[38:75],color='r')
pyplot.legend(['Nato','United States','Soviet Union'])
pyplot.xlabel('Year')
pyplot.ylabel('10s of Billions of Year 2000 US Dollars')


# In[ ]:


#Making error function for optimization between Nato and the Soviets without grievance factors
def h(theta):
    
    a,b= theta
    N0,S0 = nato[38],soviets[38]
    y0 = N0,S0
    sse = 0.0
    t = range(0,37)
    
    def R_model(y,t):
        N,S = y
        dNdt = a*S
        dSdt = b*N
        return(dNdt,dSdt)
    
    fitted = integrate.odeint(R_model,y0,t)
    
    for i in range(0,37):
        sse+=(nato[38:75][i]-fitted[i,0])**2
    
    return(sse)

optimize.minimize(h,([1,1]),method='Nelder-Mead')


# In[ ]:


#Plotting lines of best fit from optimization without grievance factors for Nato and the Soviets
def model_vitality(): 

    t = range(0,37)
    
    def dN_dt(N,S,t): 
      return(a*S)

    def dS_dt(N,S,t):
      return(b*N)

    
    # coefficients. 
    a,b = 2.23107358e-02,  1.98240260e-02
    
    def dM_dt(M,t):
      N,S = M
      return([dN_dt(N,S,t),
              dS_dt(N,S,t),])
    Mo = [nato[38],soviets[38]] # initial condition. 
    Mfit = integrate.odeint(dM_dt,Mo,t) # actual solution 

    pyplot.plot(ty[38:75],Mfit[:,0],color='b')
    pyplot.plot(ty[38:75],Mfit[:,1],color='r')
    return()
model_vitality()
pyplot.scatter(ty[38:75],soviets[38:75],color='r')
pyplot.scatter(ty[38:75],nato[38:75],color='b')
pyplot.legend(['Nato','Soviet Union'])
pyplot.xlabel('Year')
pyplot.ylabel('10s of Billions of Year 2000 US Dollars')


# In[ ]:


#Plotting lines of best fit from optimization without grievance factors for Nato and the Soviets
def model_vitality(): 

    t = range(0,37)
    
    def dN_dt(N,S,t): 
      return(a*S)

    def dS_dt(N,S,t):
      return(b*N)

    
    # coefficients. 
    a,b = 1.23107358e-02,  2.78240260e-02
    
    def dM_dt(M,t):
      N,S = M
      return([dN_dt(N,S,t),
              dS_dt(N,S,t),])
    Mo = [murica[38],soviets[38]] # initial condition. 
    Mfit = integrate.odeint(dM_dt,Mo,t) # actual solution 

    pyplot.plot(ty[38:75],Mfit[:,0],color='g')
    pyplot.plot(ty[38:75],Mfit[:,1],color='r')
    return()
model_vitality()
pyplot.scatter(ty[38:75],soviets[38:75],color='r')
pyplot.scatter(ty[38:75],murica[38:75],color='g')
pyplot.legend(['United States','Soviet Union'])
pyplot.xlabel('Year')
pyplot.ylabel('10s of Billions of Year 2000 US Dollars')


# In[ ]:





# In[ ]:


#Making error function for optimization between Nato and the Soviets without grievance factors
def h(theta):
    
    a,c,b,d= theta
    N0,S0 = nato[38],soviets[38]
    y0 = N0,S0
    sse = 0.0
    t = range(0,37)
    
    def R_model(y,t):
        N,S = y
        dNdt = a*S - c*N
        dSdt = b*N - d*S
        return(dNdt,dSdt)
    
    fitted = integrate.odeint(R_model,y0,t)
    
    for i in range(0,37):
        sse+=(nato[38:75][i]-fitted[i,0])**2
    
    return(sse)

optimize.minimize(h,([1,1,1,1]))
    


# In[ ]:


#Plotting lines of best fit from optimization without grievance factors for Nato and the Soviets
def model_vitality(): 

    t = range(0,37)
    
    def dN_dt(N,S,t): 
      return(a*S - c*N)

    def dS_dt(N,S,t):
      return(b*N - d*S)

    
    # coefficients. 
    a,c,b,d = 0.29826182, 0.18455534, 0.17612598, 0.23578116
    
    def dM_dt(M,t):
      N,S = M
      return([dN_dt(N,S,t),
              dS_dt(N,S,t),])
    Mo = [nato[38],soviets[38]] # initial condition. 
    Mfit = integrate.odeint(dM_dt,Mo,t) # actual solution 

    pyplot.plot(ty[38:75],Mfit[:,0],color='b')
    pyplot.plot(ty[38:75],Mfit[:,1],color='r')
    return()
model_vitality()
pyplot.scatter(ty[38:75],soviets[38:75],color='r')
pyplot.scatter(ty[38:75],nato[38:75],color='b')
pyplot.legend(['Nato','Soviet Union'])
pyplot.xlabel('Year')
pyplot.ylabel('10s of Billions of Year 2000 US Dollars')


# In[ ]:


#Making error function for optimization between Nato and the Soviets with grievance factors
def h(theta):
    
    a,c,b,d,g,r= theta
    N0,S0 = nato[38],soviets[38]
    y0 = N0,S0
    sse = 0.0
    t = range(0,37)
    
    def R_model(y,t):
        N,S = y
        dNdt = a*S - c*N + g
        dSdt = b*N - d*S + r
        return(dNdt,dSdt)
    
    fitted = integrate.odeint(R_model,y0,t)
    
    for i in range(0,37):
        sse+=(nato[38:75][i]-fitted[i,0])**2
    
    return(sse)

optimize.minimize(h,([1,1,1,1,1,1]))


# In[ ]:


#Plotting lines of best fit from optimization with grievance factors for Nato and the Soviets
def model_vitality(): 

    t = range(0,37)
    
    def dN_dt(N,S,t): 
      return(a*S - c*N + g)

    def dS_dt(N,S,t):
      return(b*N - d*S + r)

    
    # coefficients. 
    a,c,b,d,g,r =0.60335855,  0.04883715,  0.0956054 ,  0.59084156, -8.30366162, 7.15156696
    
    def dM_dt(M,t):
      N,S = M
      return([dN_dt(N,S,t),
              dS_dt(N,S,t),])
    Mo = [nato[38],soviets[38]] # initial condition. 
    Mfit = integrate.odeint(dM_dt,Mo,t) # actual solution 

    pyplot.plot(ty[38:75],Mfit[:,0],color='b')
    pyplot.plot(ty[38:75],Mfit[:,1],color='r')
    return()
model_vitality()
pyplot.scatter(ty[38:75],soviets[38:75],color='r')
pyplot.scatter(ty[38:75],nato[38:75],color='b')
pyplot.legend(['Nato','Soviet Union'])
pyplot.xlabel('Year')
pyplot.ylabel('10s of Billions of Year 2000 US Dollars')


# In[ ]:


#Making error function for optimization between the US and the Soviets without grievance factors
def h(theta):
    
    a,c,b,d= theta
    N0,S0 = murica[38],soviets[38]
    y0 = N0,S0
    sse = 0.0
    t = range(0,37)
    
    def R_model(y,t):
        N,S = y
        dNdt = a*S - c*N
        dSdt = b*N - d*S
        return(dNdt,dSdt)
    
    fitted = integrate.odeint(R_model,y0,t)
    
    for i in range(0,37):
        sse+=(murica[38:75][i]-fitted[i,0])**2
    
    return(sse)

optimize.minimize(h,([1,1,1,1]))


# In[ ]:


#Plotting lines of best fit from optimization without grievance factors for the US and the Soviets
def model_vitality(): 

    t = range(0,37)
    
    def dN_dt(N,S,t): 
      return(a*S - c*N)

    def dS_dt(N,S,t):
      return(b*N - d*S)

    
    # coefficients. 
    a,c,b,d = 0.17873619, 0.16582989, 0.20141431, 0.18168638
    
    def dM_dt(M,t):
      N,S = M
      return([dN_dt(N,S,t),
              dS_dt(N,S,t),])
    Mo = [murica[38],soviets[38]] # initial condition. 
    Mfit = integrate.odeint(dM_dt,Mo,t) # actual solution 

    pyplot.plot(ty[38:75],Mfit[:,0],color='g')
    pyplot.plot(ty[38:75],Mfit[:,1],color='r')
    return()
model_vitality()
pyplot.scatter(ty[38:75],soviets[38:75],color='r')
pyplot.scatter(ty[38:75],murica[38:75],color='g')
pyplot.legend(['United States','Soviet Union'])
pyplot.xlabel('Year')
pyplot.ylabel('10s of Billions of Year 2000 US Dollars')


# In[ ]:


#Making error function for optimization between the US and the Soviets with grievance factors
def h(theta):
    
    a,c,b,d,g,r= theta
    N0,S0 = murica[38],soviets[38]
    y0 = N0,S0
    sse = 0.0
    t = range(0,37)
    
    def R_model(y,t):
        N,S = y
        dNdt = a*S - c*N + g
        dSdt = b*N - d*S + r
        return(dNdt,dSdt)
    
    fitted = integrate.odeint(R_model,y0,t)
    
    for i in range(0,37):
        sse+=(murica[38:75][i]-fitted[i,0])**2
    
    return(sse)

optimize.minimize(h,([1,1,1,1,1,1]))


# In[ ]:


#Plotting lines of best fit from optimization with grievance factors for the US and the Soviets
def model_vitality(): 

    t = range(0,37)
    
    def dN_dt(N,S,t): 
      return(a*S - c*N + g)

    def dS_dt(N,S,t):
      return(b*N - d*S + r)

    
    # coefficients. 
    a,c,b,d,g,r =0.09975891,  -0.01540524,   1.01118386,   0.53496123, -2.70424892, -12.29044219
    
    def dM_dt(M,t):
      N,S = M
      return([dN_dt(N,S,t),
              dS_dt(N,S,t),])
    Mo = [murica[38],soviets[38]] # initial condition. 
    Mfit = integrate.odeint(dM_dt,Mo,t) # actual solution 

    pyplot.plot(ty[38:75],Mfit[:,0],color='g')
    pyplot.plot(ty[38:75],Mfit[:,1],color='r')
    return()
model_vitality()
pyplot.scatter(ty[38:75],soviets[38:75],color='r')
pyplot.scatter(ty[38:75],murica[38:75],color='g')
pyplot.legend(['United States','Soviet Union'])
pyplot.xlabel('Year')
pyplot.ylabel('10s of Billions of Year 2000 US Dollars')


# In[ ]:


#Predicting 2007 for NATO VS Soviet Union (without collapse of Soviet Union) without grievance factors
#The optimization for lines of best fit was done only with data before collapse, its interesting that the data after still matches
def model_vitality(): 

    t = range(0,56)
    
    def dN_dt(N,S,t): 
      return(a*S - c*N)

    def dS_dt(N,S,t):
      return(b*N - d*S)

    
    # coefficients. 
    a,c,b,d = 0.29826182, 0.18455534, 0.17612598, 0.23578116
    
    def dM_dt(M,t):
      N,S = M
      return([dN_dt(N,S,t),
              dS_dt(N,S,t),])
    Mo = [nato[38],soviets[38]] # initial condition. 
    Mfit = integrate.odeint(dM_dt,Mo,t) # actual solution 

    pyplot.plot(ty[38:],Mfit[:,0],color='b')
    pyplot.plot(ty[38:],Mfit[:,1],color='r')
    return()
model_vitality()
pyplot.scatter(ty[38:94],soviets[38:94],color='r')
pyplot.scatter(ty[38:94],nato[38:94],color='b')
pyplot.legend(['Nato','Soviet Union'])
pyplot.xlabel('Year')
pyplot.ylabel('10s of Billions of Year 2000 US Dollars')


# In[ ]:


#Predicting 2007 for NATO VS Soviet Union (without collapse of Soviet Union) with grievance factors
#The optimization for lines of best fit was done only with data before collapse
def model_vitality(): 

    t = range(0,56)
    
    def dN_dt(N,S,t): 
      return(a*S - c*N + g)

    def dS_dt(N,S,t):
      return(b*N - d*S + r)

    
    # coefficients. 
    a,c,b,d,g,r =0.60335855,  0.04883715,  0.0956054 ,  0.59084156, -8.30366162, 7.15156696
    
    def dM_dt(M,t):
      N,S = M
      return([dN_dt(N,S,t),
              dS_dt(N,S,t),])
    Mo = [nato[38],soviets[38]] # initial condition. 
    Mfit = integrate.odeint(dM_dt,Mo,t) # actual solution 

    pyplot.plot(ty[38:94],Mfit[:,0],color='b')
    pyplot.plot(ty[38:94],Mfit[:,1],color='r')
    return()
model_vitality()
pyplot.scatter(ty[38:],soviets[38:],color='r')
pyplot.scatter(ty[38:],nato[38:],color='b')
pyplot.legend(['Nato','Soviet Union'])
pyplot.xlabel('Year')
pyplot.ylabel('10s of Billions of Year 2000 US Dollars')


# In[ ]:


#Predicting 2007 for US VS Soviet Union (without collapse of Soviet Union) without grievance factors
#The optimization for lines of best fit was done only with data before collapse, its interesting that the data after still matches
def model_vitality(): 

    t = range(0,56)
    
    def dN_dt(N,S,t): 
      return(a*S - c*N)

    def dS_dt(N,S,t):
      return(b*N - d*S)

    
    # coefficients. 
    a,c,b,d = 0.17873619, 0.16582989, 0.20141431, 0.18168638
    
    def dM_dt(M,t):
      N,S = M
      return([dN_dt(N,S,t),
              dS_dt(N,S,t),])
    Mo = [murica[38],soviets[38]] # initial condition. 
    Mfit = integrate.odeint(dM_dt,Mo,t) # actual solution 

    pyplot.plot(ty[38:94],Mfit[:,0],color='g')
    pyplot.plot(ty[38:94],Mfit[:,1],color='r')
    return()
model_vitality()
pyplot.scatter(ty[38:],soviets[38:],color='r')
pyplot.scatter(ty[38:],murica[38:],color='g')
pyplot.legend(['United States','Soviet Union'])
pyplot.xlabel('Year')
pyplot.ylabel('10s of Billions of Year 2000 US Dollars')


# In[ ]:


#Predicting 2007 for US VS Soviet Union (without collapse of Soviet Union) with grievance factors
#The optimization for lines of best fit was done only with data before collapse
def model_vitality(): 

    t = range(0,56)
    
    def dN_dt(N,S,t): 
      return(a*S - c*N + g)

    def dS_dt(N,S,t):
      return(b*N - d*S + r)

    
    # coefficients. 
    a,c,b,d,g,r =0.09975891,  -0.01540524,   1.01118386,   0.53496123, -2.70424892, -12.29044219
    
    def dM_dt(M,t):
      N,S = M
      return([dN_dt(N,S,t),
              dS_dt(N,S,t),])
    Mo = [murica[38],soviets[38]] # initial condition. 
    Mfit = integrate.odeint(dM_dt,Mo,t) # actual solution 

    pyplot.plot(ty[38:94],Mfit[:,0],color='g')
    pyplot.plot(ty[38:94],Mfit[:,1],color='r')
    return()
model_vitality()
pyplot.scatter(ty[38:],soviets[38:],color='r')
pyplot.scatter(ty[38:],murica[38:],color='g')
pyplot.legend(['United States','Soviet Union'])
pyplot.xlabel('Year')
pyplot.ylabel('10s of Billions of Year 2000 US Dollars')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




