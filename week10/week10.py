from pylab import *
import scipy.signal as sp 
import numpy as np 
with open("h.csv","r") as f:
	lines = f.readlines()
fil = []
for i in range(0,len(lines)): fil.append(float(lines[i].strip())) 
fil_w,fil_h = sp.freqz((fil))
fig,ax = plt.subplots(2,sharex=True)
ax[0].plot(fil_w,abs(fil_h),"b")
ax[1].plot(fil_w,angle(fil_h))
plt.show()
n = np.linspace(1,1024,1024)
x = cos(0.2*pi*n) + cos(0.85*pi*n)
y = np.convolve(x,np.array(fil))
fig2,bx = plt.subplots(1)
bx.plot(n[:128],y[:128])
show()
z = ifft(fft(x)*fft(np.concatenate((fil,np.zeros(len(x)-len(fil))))))
fig3,cx = plt.subplots(1)
cx.plot(n[:128],z[:128])
show()
fil_padded = np.concatenate((fil,np.zeros((16-len(fil)))))
x_padded = np.concatenate((np.zeros((11)),x))
iters = len(x)/len(fil_padded)
w=[]
for i in range(0,int(iters)):
	x_temp = x_padded[i*16:(i*16)+27]
	h_temp = np.concatenate((fil_padded,np.zeros((len(x_temp)-len(fil_padded)))))
	z_temp = ifft(fft(x_temp)*fft(h_temp))
	w.append(z_temp[11:])
w = (np.asarray(w)).flatten()
fig3,cx = plt.subplots(1)
cx.plot(n[:128],w[:128])
show()
with open("x1.csv","r") as f: lines = f.readlines()
cheff=[]
for i in range(len(lines)) : cheff.append(complex(lines[i].strip().replace("i","j")))
cheff_w,cheff_h = sp.freqz(cheff)
fig4,ex = plt.subplots(2,sharex=True)
ex[0].plot(cheff_w,abs(cheff_h),"b")
ex[1].plot(cheff_w,unwrap(angle(cheff_h)))
plt.show()
cheff_shifted = roll(cheff,5)
peak = ifft(conj(fft(cheff))*(fft(cheff_shifted)))
plot(arange(len(cheff)),peak)
xlim(2,8)
show()