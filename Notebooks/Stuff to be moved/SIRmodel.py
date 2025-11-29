#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
#%%
def sir_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta*S*I
    dIdt = beta*S*I-gamma*I
    dRdt = gamma*I

    return [dSdt,dIdt,dRdt]

y0 = [1-0.001,0.001,0] #This is the initial condition to SIR model
beta = 0.8
gamma = 1/5.4
tspan = [0,30]
tmeasurements = np.linspace(0, 30, 100)
sol = solve_ivp(lambda t, y: sir_model(t, y, beta, gamma), tspan, y0, t_eval=tmeasurements)

#tmeasurements = np.linspace(0,30,100)
#%%
# Plot the results
#plt.plot(sol.t, sol.y[0], 'b', sol.t, sol.y[1], 'r', sol.t, sol.y[2], 'g')
#plt.legend(['S', 'I', 'R'])
#plt.show()

#Going from the observed values to the parameters.
#Here we will use the forward finite difference method first
#Jeg tror N er et 1, fordi det udgør en fraktion af populationen. 
b_foward = np.zeros(3*len(sol.y[0]))
b_backward = np.zeros(3*len(sol.y[0]))
b_center = np.zeros(3*len(sol.y[0]))
A_forward = np.zeros((3*len(sol.y[0]),2))
A_backward = np.zeros((3*len(sol.y[0]),2))
A_center = np.zeros((3*len(sol.y[0]),2))
delta_t = tmeasurements[-1]/len(tmeasurements)
N = len(tmeasurements)
k = 0
for i in range(len(tmeasurements)-1):
    b_foward[k] = (sol.y[0][i+1]-sol.y[0][i])/delta_t
    b_foward[k+1] = (sol.y[1][i+1]-sol.y[1][i])/delta_t
    b_foward[k+2] = (sol.y[2][i+1]-sol.y[2][i])/delta_t
    
    A_forward[k,0] = -sol.y[0][i]*sol.y[1][i]
    A_forward[k+1,0] = sol.y[0][i]*sol.y[1][i]
    A_forward[k+1,1] = -sol.y[1][i]
    A_forward[k+2,1] = sol.y[1][i]
    k += 3
k=0
for i in range(1,len(tmeasurements)):
    b_backward[k] = (sol.y[0][i]-sol.y[0][i-1])/delta_t
    b_backward[k+1] = (sol.y[1][i]-sol.y[1][i-1])/delta_t
    b_backward[k+2] = (sol.y[2][i]-sol.y[2][i-1])/delta_t
    
    A_backward[k,0] = -sol.y[0][i]*sol.y[1][i]
    A_backward[k+1,0] = sol.y[0][i]*sol.y[1][i]
    A_backward[k+1,1] = -sol.y[1][i]
    A_backward[k+2,1] = sol.y[1][i]
    k += 3
k = 0
for i in range(1,len(tmeasurements)-1):
    b_center[k] = (sol.y[0][i+1]-sol.y[0][i-1])/(2*delta_t)
    b_center[k+1] = (sol.y[1][i+1]-sol.y[1][i-1])/(2*delta_t)
    b_center[k+2] = (sol.y[2][i+1]-sol.y[2][i-1])/(2*delta_t)
    
    A_center[k,0] = -sol.y[0][i]*sol.y[1][i]
    A_center[k+1,0] = sol.y[0][i]*sol.y[1][i]
    A_center[k+1,1] = -sol.y[1][i]
    A_center[k+2,1] = sol.y[1][i]
    k += 3


AA_forward = np.dot(A_forward.T,A_forward)
AA_backward = np.dot(A_backward.T,A_backward)
AA_center = np.dot(A_center.T,A_center)
b_forward_normal = np.dot(A_forward.T,b_foward)
b_backward_normal = np.dot(A_backward.T,b_backward)
b_center_normal = np.dot(A_center.T,b_center)

beta_gamma_coef_forward = np.linalg.solve(AA_forward,b_forward_normal)
beta_gamma_coef_backward = np.linalg.solve(AA_backward,b_backward_normal)
beta_gamma_coef_center = np.linalg.solve(AA_center,b_center_normal)

print(f'True beta = {beta} and gamma = {gamma}')
print(f'using forward we get {beta_gamma_coef_forward}')
print(f'using backward we get {beta_gamma_coef_backward}')
print(f'using center we get {beta_gamma_coef_center}')


# %%

