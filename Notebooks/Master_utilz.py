import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
from pyshred import DataManager, SHRED, SHREDEngine, MLP, PI_SHRED, LSTM
from matplotlib.tri import Triangulation
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, random_split
import scipy.linalg

def load_circle_comsol_data(file_path, verbose = False):
    #REMARK... the csv file is saved in a manner without headers. Thus the first row is data, but is currently interpreted as headers.
    #So we are neglecting the first row of data here as for now. This should be fixed in the future.
    
    data = pd.read_csv(file_path)
    names = data.columns 
    X = data[names[0]].values
    Y = data[names[1]].values
    p = data[names[-1]].values
    index = X**2 + Y**2 <0.8**2 #In the data there was some noise on the edges outside the circle, so we filter those out.
    X_inner = X[index]
    Y_inner = Y[index]
    data_inner = data[names[2:]].values[index,:].T
    
    tri = Triangulation(X_inner, Y_inner)
    
    if verbose:
        print(f"Data loaded from {file_path}")
        print(f"Number of spatial points inside the circle: {data_inner.shape[1]}")
        print(f"Number of time steps: {data_inner.shape[0]}")
        
    return data_inner, tri

def generate_circle_sensors_from_tri(nr_sensors, tri,circle_fraction = [-np.pi,np.pi],scale = 0.6):
    #Which part of the unit circle to sample from
    samples = np.linspace(circle_fraction[0],circle_fraction[1]-2*np.pi/nr_sensors, nr_sensors)
    X_sample = np.cos(samples)*scale
    Y_sample = np.sin(samples)*scale
    sensor_indices = []
    for i in range(len(samples)):
        idx_temp = np.argmin((X_sample[i]-tri.x)**2+(Y_sample[i]-tri.y)**2)
        sensor_indices.append(idx_temp)
    sensors = [(sensor_indices[i],) for i in range(len(sensor_indices))]
    return sensors


def load_square_comsol_data(file_path, verbose = False):
    #REMARK... the csv file is saved in a manner without headers. Thus the first row is data, but is currently interpreted as headers.
    #So we are neglecting the first row of data here as for now. This should be fixed in the future.
    data = pd.read_csv(file_path)
    names = data.columns 
    X = data[names[0]].values
    Y = data[names[1]].values
    p = data[names[2:]].values[:,:].T
    tri = Triangulation(X, Y)
    
    if verbose:
        print(f"Data loaded from {file_path}")
        print(f"Number of spatial points inside the circle: {p.shape[1]}")
        print(f"Number of time steps: {p.shape[0]}")
    return p, tri

def generate_circle_sensors_from_tri(nr_sensors, tri,circle_fraction = [-np.pi,np.pi],scale = 0.6):
    #Which part of the unit circle to sample from
    sensor_step = abs(circle_fraction[1]-circle_fraction[0])/nr_sensors
    samples = np.linspace(circle_fraction[0],circle_fraction[1]-sensor_step, nr_sensors)
    X_sample = np.cos(samples)*scale
    Y_sample = np.sin(samples)*scale
    sensor_indices = []
    for i in range(len(samples)):
        idx_temp = np.argmin((X_sample[i]-tri.x)**2+(Y_sample[i]-tri.y)**2)
        sensor_indices.append(idx_temp)
    sensors = [(sensor_indices[i],) for i in range(len(sensor_indices))]
    return sensors


def trajectory_gen(data_list,lags,sensors):
    if not isinstance(data_list, list):
        raise TypeError('data type of data_list must be a list')
    t_dim = data_list[0].shape[0]
    x_dim = data_list[0].shape[1]
    trajectories = np.zeros((t_dim-lags,lags,len(sensors)))
    # print("trajectories shape",trajectories.shape)
    full_states = np.zeros((t_dim-lags,x_dim))
    
    for n, data_temp in enumerate(data_list):       
        for i in range(t_dim-lags):
            for j in range(len(sensors)):
                trajectories[i,:,j] = data_temp[i:i+lags,sensors[j]]
            full_states[i,:] = data_temp[i+lags,:]

        if n < 1:
            X = trajectories.copy()
            y = full_states.copy()
        
        else:
            X = np.vstack((X,trajectories))
            y = np.vstack((y,full_states))
    return X, y

def get_sensor_data(data_list, sensor_locations):
    # Expects that data_list is a list of 2D arrays with (time, space)
    if not isinstance(data_list, list):
        raise TypeError('data type of data_list must be a list')
    
    t_dim = data_list[0].shape[0]
    sensor_measurements_temp = np.zeros((t_dim, len(sensor_locations)))
    # print(sensor_measurements_temp.shape)
    for n, data_temp in enumerate(data_list):
        for j, sensor in enumerate(sensor_locations):
            sensor_measurements_temp[:, j] = data_temp[:, sensor]
        if n < 1:
           sensor_measurements = sensor_measurements_temp.copy()
        else:
           sensor_measurements = np.vstack((sensor_measurements, sensor_measurements_temp))
    return sensor_measurements


def data_prepare(data_list, lags, sensors, scaler = None):
    if  scaler is None:
        # sensor_measurements = get_sensor_data(data_list, sensors)
        # scaler_X = MinMaxScaler()
      
        # scaler_X = scaler_X.fit(sensor_measurements)
        # print(sensor_measurements.shape)
        
        X, y = trajectory_gen(data_list, lags, sensors)
              
        scaler = {
            "min": np.min(y),
            "max": np.max(y)
        }
        
        # scaler_y = scaler_y.fit(y.T)
        y = (y - scaler["min"])/(scaler["max"] - scaler["min"])
        X = (X-scaler["min"])/(scaler["max"] - scaler["min"])
        # for i in range(X.shape[0]):
        #     X[i, :, :] = scaler_X.transform(X[i, :, :].T).T
            
        return X, y, scaler
    else:
        X, y = trajectory_gen(data_list, lags, sensors)
        y = (y - scaler["min"])/(scaler["max"] - scaler["min"])
        X = (X-scaler["min"])/(scaler["max"] - scaler["min"])
        # for i in range(X.shape[0]):
        #     X[i, :, :] = scaler_X.transform(X[i, :, :].T).T
        return X, y
    

    
def SVD_pod_reconstruction(U_r,y, sensors, r):

    C= np.zeros_like(U_r.T)
    for i,sens in enumerate(sensors[:r]):
        C[i,sens] = 1
        
    A = C.dot(U_r)
    A_U, A_S, A_V = np.linalg.svd(A, full_matrices=False)
    P_inv_A = np.dot(A_V.T,np.dot(np.diag(1/A_S), A_U.T))
    
    s = y[sensors[:r]]
    a = P_inv_A.dot(s)

    recon = np.zeros_like(y)
    for i in range(r):
        recon += a[i]*U_r[:,i]
    
    return recon

class SHREDdata(torch.utils.data.Dataset):
    """
    PyTorch Dataset for time series sensor data and corresponding full-state measurements.

    Parameters
    ----------
    DATA: dictionary containg X and y
    
    X : torch.Tensor
        Input sensor sequences of shape (batch_size, lags, num_sensors).
    Y : torch.Tensor
        Target full-state measurements of shape (batch_size, state_dim).

    Attributes
    ----------
    X : torch.Tensor
        Sensor measurement sequences.
    Y : torch.Tensor
        Full-state target measurements.
    len : int
        Number of samples in the dataset.
    """

    def __init__(self, DATA):
        """
        Initialize the TimeSeriesDataset.

        Parameters
        ----------
        X : torch.Tensor
            Input sensor sequences of shape (batch_size, lags, num_sensors).
        Y : torch.Tensor
            Target full-state measurements of shape (batch_size, state_dim).
        """
        self.X = DATA['X']
        self.Y = DATA['y']
        self.len = self.X.shape[0]
        
    def __getitem__(self, index):
        """
        Get a single sample from the dataset.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            (sensor_sequence, target_state) pair.
        """
        return self.X[index], self.Y[index]

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return self.len
     
    def split_data(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42,sequential_split=False):
        if sequential_split:
            total_len = len(self)
            train_end = int(total_len * train_ratio)
            val_end = train_end + int(total_len * val_ratio)

            train_data = {
                'X': self.X[:train_end],
                'y': self.Y[:train_end]
            }
            val_data = {
                'X': self.X[train_end:val_end],
                'y': self.Y[train_end:val_end]
            }
            test_data = {
                'X': self.X[val_end:],
                'y': self.Y[val_end:]
            }
            train_dataset = SHREDdata(train_data)
            val_dataset = SHREDdata(val_data)
            test_dataset = SHREDdata(test_data)

        else:
            generator = torch.Generator().manual_seed(seed)
            train, val, test = torch.utils.data.random_split(self, [train_ratio, val_ratio, test_ratio], generator=generator)

            train_data = {
                'X':self.X[train.indices],
                'y':self.Y[train.indices]
            }
            val_data = {
                'X':self.X[val.indices],
                'y':self.Y[val.indices]
            }
            test_data = {
                'X':self.X[test.indices],
                'y':self.Y[test.indices]
            }
            train_dataset = SHREDdata(train_data)
            val_dataset = SHREDdata(val_data)
            test_dataset = SHREDdata(test_data)

        return train_dataset,val_dataset,test_dataset
    
    def move_to_device(self, device, model_dtype = torch.float32):
        self.X = torch.tensor(self.X, device=device, dtype=model_dtype)
        self.Y = torch.tensor(self.Y, device=device, dtype=model_dtype)
        return self
    

class SHREDdata_PI(torch.utils.data.Dataset):
    """
    PyTorch Dataset for time series sensor data and corresponding full-state measurements.

    Parameters
    ----------
    DATA: dictionary containg X and y
    
    X : torch.Tensor
        Input sensor sequences of shape (batch_size, lags, num_sensors).
    Y : torch.Tensor
        Target full-state measurements of shape (batch_size, state_dim).

    Attributes
    ----------
    X : torch.Tensor
        Sensor measurement sequences.
    Y : torch.Tensor
        Full-state target measurements.
    len : int
        Number of samples in the dataset.
    """

    def __init__(self, DATA):
        """
        Initialize the TimeSeriesDataset.

        Parameters
        ----------
        X : torch.Tensor
            Input sensor sequences of shape (batch_size, lags, num_sensors).
        Y : torch.Tensor
            Target full-state measurements of shape (batch_size, state_dim).
        """
        self.X = DATA['X']
        self.coords = DATA['coords']
        self.Y = DATA['y']
        self.len = self.X.shape[0]
        
    def __getitem__(self, index):
        """
        Get a single sample from the dataset.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            (sensor_sequence, target_state) pair.
        """
        return self.X[index], self.coords, self.Y[index]

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return self.len
     
    def split_data(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42,sequential_split=False):
        if sequential_split:
            total_len = len(self)
            train_end = int(total_len * train_ratio)
            val_end = train_end + int(total_len * val_ratio)

            train_data = {
                'X': self.X[:train_end],
                'y': self.Y[:train_end],
                "coords" : self.coords
            }
            val_data = {
                'X': self.X[train_end:val_end],
                'y': self.Y[train_end:val_end],
                "coords" : self.coords
            }
            test_data = {
                'X': self.X[val_end:],
                'y': self.Y[val_end:],
                "coords" : self.coords
            }
            train_dataset = SHREDdata_PI(train_data)
            val_dataset = SHREDdata_PI(val_data)
            test_dataset = SHREDdata_PI(test_data)

        else:
            generator = torch.Generator().manual_seed(seed)
            train, val, test = torch.utils.data.random_split(self, [train_ratio, val_ratio, test_ratio], generator=generator)

            train_data = {
                'X':self.X[train.indices],
                'y':self.Y[train.indices],
                "coords" : self.coords
            }
            val_data = {
                'X':self.X[val.indices],
                'y':self.Y[val.indices],
                "coords" : self.coords
            }
            test_data = {
                'X':self.X[test.indices],
                'y':self.Y[test.indices],
                "coords" : self.coords
            }
            train_dataset = SHREDdata_PI(train_data)
            val_dataset = SHREDdata_PI(val_data)
            test_dataset = SHREDdata_PI(test_data)

        return train_dataset,val_dataset,test_dataset
    
    def move_to_device(self, device, model_dtype = torch.float32):
        self.X = torch.tensor(self.X, device=device, dtype=model_dtype)
        self.coords = torch.tensor(self.coords, device=device, dtype=model_dtype)
        self.Y = torch.tensor(self.Y, device=device, dtype=model_dtype)
        return self

#Method for QR-POD method from Jan williams paper on SHRED
def qr_place(data_matrix, num_sensors):
    '''Takes a (m x N) data matrix consisting of N samples of an m dimensional state and
    number of sensors, returns QR placed sensors and U_r for the SVD X = U S V^T'''
    "REMARK THIS IS CURRENTLY COPIED FROM JAN WILLIAMS AND SHOULD NOT BE USED OTHER THAN AS A NOTEE"
    u, s, v = np.linalg.svd(data_matrix, full_matrices=False)
    rankapprox = u[:, :num_sensors]
    q, r, pivot = scipy.linalg.qr(rankapprox.T, pivoting=True)
    sensor_locs = pivot[:num_sensors]
    return sensor_locs, rankapprox