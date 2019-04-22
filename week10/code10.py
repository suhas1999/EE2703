import numpy as np
import csv
from scipy import signal
import matplotlib.pyplot as plt
from math import *
import pandas as pd
a = np.genfromtxt('h.csv',delimiter=',')
w,h = signal.freqz(a)
fig,ax = plt.subplots(2,sharex=True)
plt.grid(True,which="all")
ax[0].plot(w,(abs(h)),"b")
ax[0].set_title("Filter Magnitude response")
ax[0].set_xlabel("Frequency(w rad/s)")
ax[0].set_ylabel("AMplitude dB")
angle = np.unwrap(np.angle(h))
ax[1].plot(w,angle,"g")
ax[1].set_title("Filter Phase response")
ax[1].set_xlabel("Frequency(w rad/s)")
ax[1].set_ylabel("Phase")
plt.show()

n = np.linspace(1,2**10,2**10)
x = np.cos(0.2*pi*n) + np.cos(0.85*pi*n)
fig2,bx = plt.subplots(1,sharex=True)
bx.plot(n,x)
bx.set_title("Sequence plot")
bx.set_xlabel("n")
bx.set_ylabel("x")
bx.set_xlim(0,40)
plt.show()
y = np.convolve(x,a,mode="same")
fig3,cx = plt.subplots(1,sharex=True)
cx.plot(y)
cx.set_title("Filtered output plot using linear convolution ")
cx.set_xlabel("n")
cx.set_ylabel("y")
cx.set_xlim(0,40)
plt.show()
#We observed that acted as a low pass filter!!!



# second method
a_adjusted = np.pad(a,(0,len(x)-len(a)),"constant")
y1 = np.fft.ifft(np.fft.fft(x) * np.fft.fft(a_adjusted))
fig4,dx = plt.subplots(1,sharex=True)
dx.plot(y1)
dx.set_title("Filtered output plot using circular convolution ")
dx.set_xlabel("n")
dx.set_ylabel("y")
dx.set_xlim(0,40)
plt.show()

# third method!!
N = len(a) + len(x) - 1
fil = np.concatenate([a,np.zeros(N-len(a))])
y_modified = np.concatenate([x,np.zeros(N-len(x))])
y2 = np.fft.ifft(np.fft.fft(y_modified) * np.fft.fft(fil))
fig5,fx = plt.subplots(1,sharex=True)
fx.plot(y2)
fx.set_title("Filtered output plot using linear convolution as circular convolution ")
fx.set_xlabel("n")
fx.set_ylabel("y")
fx.set_xlim(0,40)
plt.show()



###
zadoff = pd.read_csv("x1.csv").values[:,0]
zadoff = np.array([complex(zadoff[i].replace("i","j")) for i in range(len(zadoff))])
zw,zh = signal.freqz(zadoff)
fig5,ex = plt.subplots(2,sharex=True)
plt.grid(True,which="all")
ex[0].plot(zw,(abs(zh)),"b")
ex[0].set_title("zadoff Magnitude response")
ex[0].set_xlabel("Frequency(w rad/s)")
ex[0].set_ylabel("Zadoff Amplitude dB")
angle_z = np.unwrap(np.angle(zh))
ex[1].plot(zw,angle_z,"g")
ex[1].set_title("Zadoff Phase response")
ex[1].set_xlabel("Frequency(w rad/s)")
ex[1].set_ylabel("Phase")
plt.show()

zadoff_modified = np.concatenate([zadoff[-5:],zadoff[:-5]])

z_out = np.correlate(zadoff,zadoff_modified,"full")
fig7,gx = plt.subplots(1,sharex=True)
plt.grid(True,which="all")
gx.plot((abs(z_out)),"b")
gx.set_title(" correlation of zadoff  and shifted Z Magnitude ")
gx.set_xlabel("n")
gx.set_ylabel("Magnitude")
plt.show()



