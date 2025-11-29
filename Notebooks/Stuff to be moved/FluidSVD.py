#%%
import numpy as np    
import scipy.io as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#%% Loading in the data
fluid_data = sp.loadmat('cylALL.mat')
#%% Checking what keys there is in the data
print(fluid_data.keys())
#p = 1

#%% Withdrawing the data that is needed
velocity_data = fluid_data['UALL']
test = velocity_data[:,3]
n = int(fluid_data['n'][0][0])
m = int(fluid_data['m'][0][0])
#%%Calculating the SVD in exercise 1.7 a)
Uhat,Shat,Vhat = np.linalg.svd(velocity_data, full_matrices=False)
Sdiag = np.diag(Shat)
eigen_flow1 = Uhat[:,1].reshape((n,m),order='C')
plt.plot(Shat,'x')
plt.yscale('log')
plt.show()
#%% 1.7 b)
eigen_flow1 = Uhat[:,0].reshape((n,m),order='C')
fig,ax = plt.subplots(figsize = (12,6))
ax.imshow(eigen_flow1.T)
plt.show()

Cum_var = np.zeros(len(Shat))
sumsum = np.sum(Shat)
Cum_var[0] = Shat[0]
for i in range(1,len(Shat)):
    Cum_var[i] = Cum_var[i-1]+Shat[i]
Cum_var = Cum_var/sumsum
for k,i in enumerate(Cum_var):
    if i < 0.999:
        r = k

#Utilde = np.dot(Uhat[:,0:r],np.dot(Sdiag[0:r,0:r],Vhat[:,0:r].T))
W = np.dot(Sdiag[0:r,0:r],Vhat[0:r,:])
#Calculating the truntacte version of our original data. We call this Utilde
Utilde = np.dot(Uhat[:,0:r],W)
#%% This displays the truncated error over every flow snapshot
error = np.zeros(len(Utilde[0,:]))
for i in range(len(Utilde[0,:])):
    error[i] = np.sum(np.abs(velocity_data[i]-Utilde[i]))
plt.plot(error)
plt.show()

#%% This plots the flow snapshots as re_flow1
re_flow1 = Utilde[:,0].reshape((n,m),order='C')
fig,ax = plt.subplots(figsize = (12,6))
ax.imshow(re_flow1.T)
plt.show()
# %% Creating an animation
# Create a figure and axis
frame0 = Utilde[:,0].reshape((n,m),order='C')
fig, ax = plt.subplots()
# Display the initial matrix
img = ax.imshow(frame0.T, animated=True)

def update1(frame):
    # Update the matrix with new values
    new_frame = Utilde[:,frame+1].reshape((n,m),order='C').T
    img.set_array(new_frame)
    return img,

ani = FuncAnimation(fig, update1, frames=range(len(Utilde[0,:])-1), blit=True)

# Display the animation
plt.ion()
plt.show()

#%%
ani.save('animation.gif', writer='imagemagick', fps=30)

#%%
r = 10
W = np.dot(Sdiag[0:r,0:r],Vhat[0:r,:])
l_data = len(W[0,:])
W0 = W[:,0:(l_data-1)]
W1 = W[:,1:l_data]

Uw, Sw, Vw = np.linalg.svd(W0, full_matrices=False)
Sw_inv = np.diag(1/Sw)
W0_pseudo_inv = np.dot(Vw.T,np.dot(Sw_inv,Uw.T))

A = np.dot(W1,W0_pseudo_inv)
eigenvals_A = np.linalg.eigvals(A)
# extract real part 
Re = [ele.real for ele in eigenvals_A] 
# extract imaginary part 
Im = [ele.imag for ele in eigenvals_A] 

# %% 
plt.plot(Re, Im,'x') 
plt.ylabel('Imaginary') 
plt.xlabel('Real') 
plt.show() 

# %%
k = r
W_est = np.zeros((k,151))
W_temp = W[:,0]
W_est[:,0] = W_temp
for i in range(150):
    W_temp = np.dot(A,W_temp)
    W_est[:,i+1] = W_temp
Flow_reconstructed2 = np.dot(Uhat[:,0:k],W_est)


# %%
re_test = Flow_reconstructed2[:,0].reshape((n,m),order='C')
fig,ax = plt.subplots(figsize = (12,6))
ax.imshow(re_test.T)
plt.show()



#%%
print(np.max(np.abs(velocity_data[:,0])))
print(np.max(np.abs(Flow_reconstructed2[:,0])))
error = np.zeros(len(Flow_reconstructed2[0,:]))
for i in range(150):
    error[i] = np.max(np.abs(velocity_data[:,i]-Flow_reconstructed2[:,i]))
plt.plot(error)
plt.show()



# %%
frame0 = Flow_reconstructed2[:,2].reshape((n,m),order='C')
fig, ax = plt.subplots()
# Display the initial matrix
img = ax.imshow(frame0.T, animated=True)

def update(frame):
    # Update the matrix with new values
    new_frame = Flow_reconstructed2[:,frame+1].reshape((n,m),order='C').T
    img.set_array(new_frame)
    return img,

ani = FuncAnimation(fig, update, frames=range(len(Utilde[0,:])-10), blit=True)

# Display the animation


ani.save('animation_reconstructed.gif', writer='imagemagick', fps=10)
# %%
