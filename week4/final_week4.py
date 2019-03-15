
# coding: utf-8

# In[1]:


# Importing numpy and matplotlib
from pylab import *
from scipy.integrate import quad


# In[2]:


rcParams['figure.figsize'] = 12,9
rcParams['font.size'] = 18
rcParams['text.usetex'] = True


# In[3]:


def periodic(a, b):
    interval = b - a
    return lambda f: lambda x: f((x - a) % interval + a)


# In[7]:


def coscos(x):
    return cos(cos(x))
def per_e(x):
    return exp(np.remainder(x,2*pi))


# In[8]:


x = linspace(-2*pi,4*pi,400)
fig1 = figure()
ax = fig1.add_subplot(1,1,1)
ax.set_yscale('log')
grid(True)
plot(x,exp(x))
plot(x,per_e(x))
title("$e^x$ and its periodically extended version")
ylabel(r"$y$ (log)", fontsize = 26)
xlabel(r"$x$ (linear)")
legend([r"$e^x$",r"periodic version of $e^x$"], loc=0)
show()


# In[27]:


fig2 = figure()
grid(True)
plot(x,cos(cos(x)))
plot(x,coscos(x))
title("$\cos(\cos(x))$ and its periodically extended version")
ylabel(r"$y$ (linear)", fontsize = 26)
xlabel(r"$x$ (linear)")
legend([r"$\cos(\cos(x))$",r"periodic version of $\cos(\cos(x))$"], loc=1)
ylim(0.3,1.4)
show()


# In[11]:


def quad_fourier(f,n):
    """
    Find the n even and odd fourier coefficients of f, and the DC
    value, using quad integration.
    Assumes a period from 0 to 2pi.
    """
    
    # functions to integrate
    u = lambda x,k : f(x)*cos(k*x)
    v = lambda x,k : f(x)*sin(k*x)
    
    # DC coefficient
    a0 = 1/(2*pi)*quad(f,0,2*pi)[0]
    
    # find coefficients by integrating
    ret = [a0]
    for k in arange(n)+1:
        ak = 1/pi*quad(u,0,2*pi,args=(k))[0]
        bk = 1/pi*quad(v,0,2*pi,args=(k))[0]
        ret.append(ak)
        ret.append(bk)
        
    return array(ret)


# In[12]:


ef = quad_fourier(per_e,25)
cf = quad_fourier(coscos,25)


# In[13]:


def plotCoeffs(coeffs, name="Coefficients"):
    """
    Helper function to scatter the given coefficients on a 
    semilog and loglog scale
    """
    
    fig0 = figure()
    ax = fig0.add_subplot(1,1,1)
    ax.set_yscale('log')
    grid(True)
    title("Fourier coefficients of {} (semilog)".format(name))
    ylabel(r"Magnitude of coefficients", fontsize = 26)
    xlabel(r"Index")
    scatter(arange(len(coeffs))+1,abs(coeffs),color='red',s=50)
    show()
    
    fig1 = figure()
    ax = fig1.add_subplot(1,1,1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    grid(True)
    title("Fourier coefficients of {} (log-log)".format(name))
    ylabel(r"Magnitude of coefficients", fontsize = 26)
    xlabel(r"Index")
    scatter(arange(len(coeffs))+1,abs(coeffs),color='red',s=50)
    
    return fig0,fig1


# In[14]:


fig3,fig4 = plotCoeffs(ef,r"$e^x$")


# In[15]:


fig5,fig6 = plotCoeffs(cf,"$\cos(\cos(x))$")


# In[16]:


def lstsq_fourier(f,n,steps=400):
    """
    Find the n even and odd fourier coefficients of f, and the DC
    value, using least squares estimation. Defaults to 400 steps.
    Assumes a period from 0 to 2pi.
    
    Also returns the least squares matrix
    """
    
    x=linspace(0,2*pi,steps+1)
    x=x[:-1] # drop last term to have a proper periodic integral
    b=f(x) # f has been written to take a vector
    A=zeros((steps,2*n+1)) # allocate space for A
    A[:,0]=1 # col 1 is all ones
    for k in range(1,n+1):
        A[:,2*k-1]=cos(k*x) # cos(kx) column
        A[:,2*k]=sin(k*x) # sin(kx) column
    
    return lstsq(A,b)[0], A


# In[17]:


s = 400 # number of steps for least square estimation
elc, Aexp = lstsq_fourier(per_e,25,s)
ax = fig3.axes[0]
ax.scatter(arange(len(elc))+1,abs(elc),s=50,color='green')
ax.legend(["Using quad integration", "Using least squares"], loc=3)
fig3


# In[18]:


ax = fig4.axes[0]
ax.scatter(arange(len(elc))+1,abs(elc),color='green',s=50)
ax.legend(["Using quad integration", "Using least squares"],loc=3)
fig4


# In[19]:


clc, Acos = lstsq_fourier(coscos,25,s)
ax = fig5.axes[0]
ax.scatter(arange(len(clc))+1,abs(clc),color='green',s=50)
ax.legend(["Using quad integration", "Using least squares"],loc=1)
fig5


# In[20]:



ax = fig6.axes[0]
ax.scatter(arange(len(clc))+1,abs(clc),color='green',s=50)
ax.legend(["Using quad integration", "Using least squares"],loc=6)
fig6


# In[21]:


print(max(abs(ef-elc)))


# In[22]:


print(max(abs(cf-clc)))


# In[23]:


I = linspace(0,2*pi,s)


# In[24]:



ax = fig1.axes[0]
ax.scatter(I,dot(Aexp,ef),color = 'green')
ax.scatter(I,dot(Aexp,elc),color = 'black',marker='+')
ax.set_ylim(1e-3,1e6)
ax.legend([r"$e^x$",r"periodic version of $e^x$",
           "quad estimated","Least squares estimated"], loc = 4)
fig1


# In[25]:


figure()
grid(True)
xlim(-2,7)
plot(x,per_e(x))
plot(I,dot(Aexp,ef),linewidth=3)
title("$e^x$ and its fourier approximation")
ylabel(r"$y$ (linear)", fontsize = 26)
xlabel(r"$x$ (linear)")
legend([r"Periodic version of $e^x$",r"Fourier estimate of $e^x$"], loc=0)
show()


# In[28]:


ax = fig2.axes[0]
ax.scatter(I,dot(Acos,cf),color = 'green')
ax.scatter(I,dot(Acos,clc),color = 'black')
ax.legend([r"$\cos(\cos(x))$",r"periodic version of $\cos(\cos(x))$"
           ,"quad estimated","Least squares estimated"], loc = 0)
fig2

