import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
from pyshred import DataManager, SHRED, SHREDEngine, MLP, PI_SHRED, LSTM, LSTM_PI, PI_SHRED_V2, MLP_TRUNK
from torch.utils.data import DataLoader, random_split
import Master_utilz as utilz
from scipy.fft import fft, ifft, fftshift, fftfreq
from io import BytesIO
from PIL import Image

plt.rcParams.update({
    "font.family": "serif",
    # "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",       # Computer Modern for math
    "axes.unicode_minus": False,    # So minus sign appears correctly
})

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folder_path = "Burgers_viz/"

lags = 15
nr_sensors = 3
nr_t_steps = 200
model_dtype = torch.float32
batch_size = 128

make_gif = True
#Training skips for modified SHRED
training_skips = 50


#Creating data by solving the burgers equation in 1D
def init_KS(x,t=0):
    u = (1+np.sin(x/10))
    return u

def Fourier_n(u, Omega,n):
    u_hat = (1j*Omega)**n*fft(u)
    u_hat = ifft(u_hat)
    return u_hat

N = 500
L = 2
x = np.linspace(-L/2, L/2, N, endpoint=False)
nu = 0.01/np.pi
n = fftfreq(N)*N
scale = 2*np.pi/L
Omega = scale*n
u0 = -np.sin(np.pi*x)

def burgers(u, Omega,nu):
    dudt = -0.5*Fourier_n(u*u, Omega, 1)+nu*Fourier_n(u, Omega, 2)
    return np.real(dudt)

def runge_kutta(u0,dudt,dt,iters, Omega,nu):
    t_all = np.zeros(iters)
    u_sol_all = np.zeros((iters,u0.size))
    u_sol = u0.copy()
    for i in range(iters):
        k1 = dudt(u_sol,Omega,nu)
        k2 = dudt(u_sol + 0.5*dt*k1,Omega,nu)
        k3 = dudt(u_sol + 0.5*dt*k2,Omega,nu)
        k4 = dudt(u_sol + dt*k3,Omega,nu)
        u_sol = u_sol + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        u_sol_all[i,:] = u_sol
        t_all[i] = dt*i
    print(f"Finished RK4 with {iters} iterations final time = {dt*iters}")
    return u_sol_all,t_all

delta_t = 2*np.abs(6 / ((N) ** 2))
t_end =  1.6037/np.pi*2.5

iters = int(t_end/delta_t)
img_test = max(u0)*2/N
u_sol,t = runge_kutta(u0,burgers,delta_t,iters, Omega,nu)

t_skips = iters//nr_t_steps
x_skips = 1

t = t[::t_skips]
u_sol = u_sol[::t_skips,::x_skips]
x_coords = x[::x_skips]
delta_t_synth = delta_t*t_skips

np.random.seed(41)
sensors  = np.random.choice(np.arange(0,u_sol.shape[1]),nr_sensors,replace = False)

X_train,y_train = utilz.trajectory_gen([u_sol],lags =lags, sensors=sensors)

# training_skips = 1
# X_train = X_train
# y_train = y_train[:,::training_skips]

Burger_data = {
    'X':X_train,
    "coords" : x_coords[:,None],
    'y':y_train,
}


Burger_dataset = utilz.SHREDdata_PI(Burger_data)
train_dataset, val_dataset, _ = Burger_dataset.split_data(train_ratio = 0.6,val_ratio=0.4, test_ratio=0.0 , sequential_split=True)

train_dataset.move_to_device(device, model_dtype=model_dtype)
val_dataset.move_to_device(device, model_dtype=model_dtype)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

for X,coords, y in train_loader:
    print("X.shape", X.shape)
    print("Coords.shape",coords.shape)
    print("y.shape", y.shape)
    # print(shred.forward(X,coords[0])[0].shape)
    break

torch.set_default_dtype(model_dtype)
sequence_model = LSTM_PI(hidden_size=32, num_layers=2)
trunk_model = MLP_TRUNK(hidden_sizes=[64,128], dropout=0.1)

jakob_shred = PI_SHRED_V2(sequence_model=sequence_model,
    trunk_model=trunk_model)

jakob_shred.sequence_model.initialize(input_size=X_train.shape[2], lags=lags, decoder_type="MLP_PI",dtype = model_dtype)
jakob_shred.trunk_model.initialize(input_size=1, output_size=jakob_shred.sequence_model.hidden_size)
jakob_shred.to(device, dtype=model_dtype)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(jakob_shred.parameters(), lr=0.001)

jakob_shred.train()

for epoch in range(3500):    
    epoch_loss = 0.0
    for X,coords, y in train_loader:
        coords = coords[0][::training_skips,:]  
        y = y[:,::training_skips]
        optimizer.zero_grad()
        output_t, output_tm1,output_tm2 = jakob_shred.forward(X, coords)
        data_loss = criterion(output_t, y)
        loss = data_loss

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    if (epoch % 100) == 0:
        print(f"at epoch {epoch} the data loss is {torch.sum(data_loss):.7f}")

sequence_model = LSTM(hidden_size=32, num_layers=2)
decoder_model = MLP(hidden_sizes=[64,128], dropout=0.1)


SD_shred = PI_SHRED(sequence_model=sequence_model,
    decoder_model=decoder_model)

SD_shred.sequence.initialize(input_size=X_train.shape[2], lags=lags, decoder_type="MLP",dtype = model_dtype)
SD_shred.decoder.initialize(input_size=SD_shred.sequence.hidden_size, output_size=y_train.shape[1])
SD_shred.to(device, dtype=model_dtype)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(SD_shred.parameters(), lr=0.001)

SD_shred.train()
for epoch in range(3000):    
    epoch_loss = 0.0
    for X,coords, y in train_loader:
        coords = coords[0]  
        optimizer.zero_grad()
        output_t = SD_shred.forward(X, coords)
        data_loss = criterion(output_t, y)
        loss = data_loss

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    if (epoch % 100) == 0:
        print(f"at epoch {epoch} the data loss is {torch.sum(data_loss):.7f}")

visualize_loader = val_loader
index = int(1*len(visualize_loader)-1)
counter = 0
jakob_shred.eval()
SD_shred.eval()
for X, coords, y in visualize_loader:
    if counter == index:
        coords = coords[0]
        output_t_jakob, _, _ = jakob_shred.forward(X, coords)
        output_t_SD = SD_shred.forward(X)
        plt.figure(figsize=(10,6))
        plt.plot(coords.cpu().numpy(), output_t_SD[0].detach().cpu().numpy(), label='SHRED', linewidth=2.5)
        plt.plot(coords.cpu().numpy(), output_t_jakob[0].detach().cpu().numpy(), label='Modified SHRED', linewidth=2.5)
        plt.plot(coords.cpu().numpy(), y[0].cpu().numpy(), '--', label='True', linewidth=2)
        if training_skips >= 20:
            plt.plot(coords[::training_skips,:] .cpu().numpy(), y[0,::training_skips].cpu().numpy(), 'o', color = "black",label='Training data used', markersize=2)
        plt.title(f'SHRED Predictions \n'
                  f"Using {nr_sensors} sensors, lags = {lags}, Nx = {N} and Nt = {nr_t_steps}\n" 
                  f"Modified model uses {(N//training_skips)/N * 100}% of points in x")
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        first_sensor = True
        for sensor in sensors:
            if first_sensor:
                plt.plot(coords[sensor].cpu().numpy(), y[0,sensor].cpu().numpy(),"o", color='red', label='Sensors',markersize=5)
                first_sensor = False
            else:
                plt.plot(coords[sensor].cpu().numpy(), y[0,sensor].cpu().numpy(),"o", color='red', markersize=5)
        plt.legend()
        plt.savefig("Burgers_viz/test_plot")
        plt.show()
        break   
    counter+= 1

if make_gif: 

    images_train = []
    visualize_loader_train = DataLoader(train_dataset, batch_size=1, shuffle=False)

    jakob_shred.eval()
    SD_shred.eval()
    for i, (X, coords, y) in enumerate(visualize_loader_train):
        print(f"starting loop {i}")
        coords = coords[0]
        output_t_jakob, _, _ = jakob_shred.forward(X, coords)
        output_t_SD = SD_shred.forward(X)
        plt.figure(figsize=(10,6))
        plt.plot(coords.cpu().numpy(), output_t_SD[0].detach().cpu().numpy(), label='SHRED', linewidth=2.5)
        plt.plot(coords.cpu().numpy(), output_t_jakob[0].detach().cpu().numpy(), label='Modified SHRED', linewidth=2.5)
        plt.plot(coords.cpu().numpy(), y[0].cpu().numpy(), '--', label='True', linewidth=2)
        if training_skips >= 10:
                plt.plot(coords[::training_skips,:] .cpu().numpy(), y[0,::training_skips].cpu().numpy(), 'o', color = "black",label='Training data used', markersize=2)
        # plt.plot(coords[240:250,:] .cpu().numpy(), y[0,240:250].cpu().numpy(), 'o', color = "black",label='Training data used', markersize=2)
        plt.title(f'SHRED Predictions on train data at t = {t[i]:.4f} \n'
                  f"Using {nr_sensors} sensors, lags = {lags}, Nx = {N} and Nt = {nr_t_steps}\n" 
                  f"Modified model uses {(N//training_skips)/N * 100}% of points in x")
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        first_sensor = True
        for sensor in sensors:
            if first_sensor:
                plt.plot(coords[sensor].cpu().numpy(), y[0,sensor].cpu().numpy(),"o", color='red', label='Sensors',markersize=5)
                first_sensor = False
            else:
                plt.plot(coords[sensor].cpu().numpy(), y[0,sensor].cpu().numpy(),"o", color='red', markersize=5)
        plt.legend()
        plt.grid()
        plt.xlim([-L/2, L/2])
        plt.ylim([-1.2, 1.2])
        buf = BytesIO()
        plt.savefig(buf, format = 'png')
        buf.seek(0)
        images_train.append(Image.open(buf))
        plt.close()
        plt.show()

    save_path = folder_path + f"Train_s{nr_sensors}_xskips{training_skips}_endtime{t_end:.2f}.gif"

    images_train[0].save(save_path,
                save_all=True,
                append_images=images_train[1:],
                duration=200,
                loop=0)

    images_test = []
    visualize_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    jakob_shred.eval()
    SD_shred.eval()
    for i, (X, coords, y) in enumerate(visualize_loader):
        print(f"starting loop {i}")
        coords = coords[0]
        output_t_jakob, _, _ = jakob_shred.forward(X, coords)
        output_t_SD = SD_shred.forward(X)
        plt.figure(figsize=(10,6))
        plt.plot(coords.cpu().numpy(), output_t_SD[0].detach().cpu().numpy(), label='SHRED', linewidth=2.5)
        plt.plot(coords.cpu().numpy(), output_t_jakob[0].detach().cpu().numpy(), label='Modified SHRED', linewidth=2.5)
        plt.plot(coords.cpu().numpy(), y[0].cpu().numpy(), '--', label='True', linewidth=2)
        if training_skips >= 10:
                plt.plot(coords[::training_skips,:] .cpu().numpy(), y[0,::training_skips].cpu().numpy(), 'o', color = "black",label='Training data used', markersize=2)
        # plt.plot(coords[240:250,:] .cpu().numpy(), y[0,240:250].cpu().numpy(), 'o', color = "black",label='Training data used', markersize=2)
        plt.title(f'SHRED Predictions on test data at t = {t[len(train_dataset)+i]:.4f} \n'
                  f"Using {nr_sensors} sensors, lags = {lags}, Nx = {N} and Nt = {nr_t_steps}\n" 
                  f"Modified model uses {(N//training_skips)/N * 100}% of points in x")
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        first_sensor = True
        for sensor in sensors:
            if first_sensor:
                plt.plot(coords[sensor].cpu().numpy(), y[0,sensor].cpu().numpy(),"o", color='red', label='Sensors',markersize=5)
                first_sensor = False
            else:
                plt.plot(coords[sensor].cpu().numpy(), y[0,sensor].cpu().numpy(),"o", color='red', markersize=5)
        plt.legend()
        plt.grid()
        plt.xlim([-L/2, L/2])
        plt.ylim([-1.2, 1.2])
        buf = BytesIO()
        plt.savefig(buf, format = 'png')
        buf.seek(0)
        images_test.append(Image.open(buf))
        plt.close()
        plt.show()

    save_path = folder_path + f"Test_s{nr_sensors}_xskips{training_skips}_endtime{t_end:.2f}.gif"

    images_test[0].save(save_path,
                save_all=True,
                append_images=images_test[1:],
                duration=200,
                loop=0)

print("BING BONG!")