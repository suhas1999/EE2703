import pandas as pd
import numpy as np
from pylab import *
import scipy.special as sp
from scipy.linalg import lstsq
try:
	data = np.loadtxt("fitting.dat")       #put try else here
except IOError:
    print('Keep the fitting.dat file in same folder')
    exit()
sigma=np.logspace(-1,-3,9)

N,k = data.shape

t = np.linspace(0,10,N)

def g(t,A,B):
    return A*sp.jv(2,t) + B*t

import matplotlib.pyplot as plt
for i in range(k-1):
    plot(data[:,0],data[:,i+1],label = '$\sigma$' +"="+ str(np.around(sigma[i],3)))

plt.legend()
xlabel(r'$t$',size=20)
ylabel(r'$f(t)+n$',size=20)
title(r'Q4:Plot of the data to be fitted')
grid(True)

plot(t,g(t,1.05,-0.105),label = r"True value")
plt.legend()
show()


errorbar(t[::5],data[::5,1],sigma[1],fmt='ro',label = r"Error bar")
xlabel(r"$t$",size=20)
title(r"Q5:Data points for $\sigma$ = 0.1 along with exact function")
plt.legend()
plot(t,g(t,1.05,-0.105),label = r"True value")
plt.legend()
show()

M = np.zeros((N,2))
for i in range(N):
    M[i,0] = sp.jv(2,data[i,0])
    M[i,1] = data[i,0]

A = linspace(0,2,20)
B = linspace(-0.2,0,20)

fk = g(data[:,0],1.05,-0.105)

epsilon = np.zeros((len(A),len(B)))

for i in range(len(A)):
    for j in range((len(B))):
        epsilon[i,j] = np.mean(np.square(fk - g(t,A[i],B[j])))

cp = plt.contour(A,B,epsilon,20)
plot(1.05,-0.105,"ro")
annotate(r"$Exact\ location$",xy=(1.05,-0.105))
plt.clabel(cp,inline=True)
plt.xlabel(r"$A$",size=20)
plt.ylabel(r"$B$",size=20)
plt.title(r"Q8:Countour plot for $\epsilon_{ij}$")
show()       


pred=[]
Aerror=[]
Berror=[]
y_true = g(t,1.05,-0.105)
for i in range(k-1):
    p,resid,rank,sig=lstsq(M,data[:,i+1])
    aerr = np.square(p[0]-1.05)
    ber = np.square(p[1]+0.105)   
    Aerror.append(aerr)
    Berror.append(ber)

plot(logspace(-1,-3,9),Aerror,"ro",linestyle="--", linewidth = 1,label=r"$Aerr$")
plt.legend()
plot(logspace(-1,-3,9),Berror,"go",linestyle="--",linewidth = 1,label=r"Berr")
plt.legend()
grid(True)
plt.xlabel(r"Noise standard deviation")
plt.ylabel(r"$MS Error$",size=20)
plt.title("$Q10:Variation\ of\  error\  with\  noise$")
show()

plt.loglog(logspace(-1,-3,9),Aerror,"ro")
plt.errorbar(logspace(-1,-3,9),Aerror,np.std(Aerror),fmt="ro",label=r"$Aerr$")
plt.legend()
plt.loglog(logspace(-1,-3,9),Berror,"go")
plt.errorbar(logspace(-1,-3,9),Berror,np.std(Berror),fmt="go",label=r"$Berr$")
plt.legend()
grid(True)
plt.ylabel(r"$MS Error$",size=20)
plt.title(r"$Q10:Variation\ of\ error\ with\ noise$")
plt.xlabel(r"$\sigma_{n}$",size=20)
show()

pred=[]
error=[]
y_true = g(t,1.05,-0.105)
for i in range(k-1):
    p,resid,rank,sig=lstsq(M,data[:,i+1])
    y_pred = np.dot(M,p)
    err = np.mean(np.square(y_true-y_pred))   
    error.append(err)
plt.loglog(logspace(-1,-3,9),error,"ro")
plt.ylabel(r"$MS Error$",size=20)
plt.title("Q10:Variation of mean square error of the predicted data points",size=14)
plt.xlabel(r"$\sigma_{n}$",size=20)
show()






















