import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
from pyshred import DataManager, SHRED, SHREDEngine, MLP, PI_SHRED, LSTM, PI_SHRED_V2
from matplotlib.tri import Triangulation
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, random_split
import Master_utilz as utilz
from io import BytesIO
from PIL import Image


#Setting up cuda stuff Maybe this needs to be changed for bjobs compared to running on an interactive node
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Loading the data for traning
folder_path = "playground/"
counter = 0
data_loaded = []
skips = 1
print("Loading training data...")

for file in os.listdir(folder_path):
    if file.endswith(".csv") and file.startswith("train"):
        if counter == 0: 
            path = os.path.join(folder_path, file)
            data_temp, tri = utilz.load_square_comsol_data(path, verbose=True)
            data_temp = data_temp[0::skips,:]
            data_loaded.append(data_temp)
            counter += 1
            
        else:
            path = os.path.join(folder_path, file)
            data_temp, _ = utilz.load_square_comsol_data(path, verbose=True)
            data_temp = data_temp[0::skips,:]
            data_loaded.append(data_temp)

test_data = []
print("Loading test data...")
for file in os.listdir(folder_path):
    if file.endswith(".csv") and file.startswith("test"):
        path = os.path.join(folder_path, file)
        test_data_temp, _ = utilz.load_square_comsol_data(path, verbose=True)
        test_data_temp = test_data_temp[0::skips,:]
        test_data.append(test_data_temp)

nr_sensors_list = [6,8,10,32,64]

np.random.seed(42)
sensors_all = np.random.choice(np.arange(0,data_temp.shape[1]),nr_sensors_list[-1],replace = False)

for nr_sensors in nr_sensors_list:
    print(f"Training with {nr_sensors} sensors...")     
    
    #Parameter setup
    lags = 300
    model_dtype = torch.float32
    lr = 0.001
    sequence_model = LSTM(hidden_size=64,num_layers=3)
    decoder_model = MLP(hidden_sizes=[64,128], dropout=0.0)
    #End parameters
    
    
    _, U_r = utilz.qr_place(data_loaded[0].T,nr_sensors)
    sensors = sensors_all[:nr_sensors]
    print(f"WE ARE USING {len(sensors)} NUMBER OF SENSORS")
    #Creating traing and test datasets
    X_train, y_train, scaler = utilz.data_prepare(data_loaded, lags, sensors)
    train_data_dict = {'X':X_train,'y':y_train,'scaler':scaler}

    #Test dataset i created from a seperate data set such that we are guaranteed no data leakage
    X_test, y_test = utilz.data_prepare(test_data, lags, sensors, scaler=scaler)
    test_data_dict = {'X':X_test,'y':y_test,'scaler':scaler}

    train_data = utilz.SHREDdata(train_data_dict)
    test_dataset = utilz.SHREDdata(test_data_dict)
    train_dataset, val_dataset, _ = train_data.split_data(train_ratio = 0.8, val_ratio=0.2, test_ratio=0.0, seed=42, sequential_split=True)
    
    #Moving datasets to GPU
    train_dataset.move_to_device(device, model_dtype=model_dtype)
    val_dataset.move_to_device(device, model_dtype=model_dtype)
    # test_dataset.move_to_device(device, model_dtype=model_dtype)

    #Initialize the SHRED model
    shred = PI_SHRED_V2(sequence_model=sequence_model, decoder_model=decoder_model)

    #Training the SHRED model
    val_errors = shred.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=2000,
        batch_size= 256,
        verbose = True,
        plot = False,
        plot_modulo= 30,
        lr = lr,
        patience = 100,
        model_dtype = model_dtype
    )

    # U,S,V = np.linalg.svd(y_train.T, full_matrices=False)
    # U_r = U[:,:nr_sensors]
    
    
    test_loader = DataLoader(val_dataset, shuffle=False, batch_size=1)

    # val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1)

    svd_pred_list = []
    shred_pred_list = []
    true_sol_list = []
    error_svd_list = []
    error_shred_list = []

    for i, (X_temp,y_temp) in enumerate(test_loader):
        y_temp = y_temp.cpu().numpy()
        shred_pred = shred.forward(X_temp).detach().cpu().numpy()
        SVD_pred = utilz.SVD_pod_reconstruction(U_r, y_temp[0,:], sensors, r=nr_sensors)

        true_sol_list.append(y_temp[0,:])
        svd_pred_list.append(SVD_pred)
        shred_pred_list.append(shred_pred)

        abs_error_svd = y_temp[0,:] - SVD_pred
        abs_error_shred = y_temp[0,:] - shred_pred[0,:]
        
        error_svd_list.append(abs_error_svd)
        error_shred_list.append(abs_error_shred)
    
    nr_images = len(shred_pred_list)

    true_max = np.max(true_sol_list)
    true_min = np.min(true_sol_list)
    pred_levels = np.linspace(true_min, true_max,100)

    error_svd_hist = np.zeros(len(error_svd_list))
    error_shred_hist = np.zeros(len(svd_pred_list))
    for i in range(len(error_svd_hist)):
        error_svd_hist[i] = np.max(error_svd_list[i])
        error_shred_hist[i] = np.max(error_shred_list[i])

    svd_error_max = np.max(error_svd_list)
    svd_error_min = np.min(error_svd_list)
    svd_error_levels = np.linspace(svd_error_min,svd_error_max, 100)

    shred_error_max = np.max(error_shred_list)
    shred_error_min = np.min(error_shred_list)
    shred_error_levels = np.linspace(shred_error_min, shred_error_max,100)

    images = []
    nr_images_in_gif =np.min((len(test_loader),50))
    gif_index_increment = len(test_loader)//nr_images_in_gif
    print("Nr of images in gif", nr_images_in_gif)
    print("gif_increment",gif_index_increment)

    for i in range(nr_images):
        print(f"starting loop {i}")
        if i == nr_images_in_gif:
            break
        i = i*gif_index_increment

        fig,ax = plt.subplots(2,3,figsize=(16,8))
        plt.suptitle(f"Reconstructions using {nr_sensors} sensors with a time history of {lags} steps", fontsize=16)
        cb_true = ax[0,0].tricontourf(tri, true_sol_list[i], levels=pred_levels,cmap = "inferno")
        fig.colorbar(cb_true, ax=ax[0,0])
        ax[0,0].set_title("True solution")
        cb_shred = ax[0,1].tricontourf(tri, shred_pred_list[i][0,:], levels=pred_levels,cmap = "inferno")
        fig.colorbar(cb_shred, ax=ax[0,1])
        ax[0,1].set_title("Reconstruction with SHRED")
        cb_svd = ax[0,2].tricontourf(tri, svd_pred_list[i], levels=pred_levels,cmap = "inferno")
        fig.colorbar(cb_svd, ax=ax[0,2])
        ax[0,2].set_title("Reconstruction with SVD PODs")
        error_svd = ax[1,2].tricontourf(tri, error_svd_list[i], levels=svd_error_levels,cmap = "inferno")
        ax[1,2].set_title(r"error: $u_{true}-u_{SVD}$")
        error_shred = ax[1,1].tricontourf(tri, error_shred_list[i], levels=shred_error_levels,cmap = "inferno")
        ax[1,1].set_title(r"error: $u_{true}-u_{SHRED}$")
        fig.colorbar(error_svd, ax=ax[1,2])
        fig.colorbar(error_shred, ax=ax[1,1])
        for i in sensors:
            if i == sensors[0]:
                ax[0,0].plot(tri.x[i],tri.y[i],color = "forestgreen",marker = "o",label = 'sensors')
                ax[0,1].plot(tri.x[i],tri.y[i],color = "forestgreen",marker = "o",label = 'sensors')
                ax[0,2].plot(tri.x[i],tri.y[i],color = "forestgreen",marker = "o",label = 'sensors')
                ax[1,1].plot(tri.x[i],tri.y[i],color = "forestgreen",marker = "o",label = 'sensors')
                ax[1,2].plot(tri.x[i],tri.y[i],color = "forestgreen",marker = "o",label = 'sensors')
            else:
                ax[0,0].plot(tri.x[i],tri.y[i],color = "forestgreen",marker = "o")
                ax[0,1].plot(tri.x[i],tri.y[i],color = "forestgreen",marker = "o")
                ax[0,2].plot(tri.x[i],tri.y[i],color = "forestgreen",marker = "o")
                ax[1,1].plot(tri.x[i],tri.y[i],color = "forestgreen",marker = "o")
                ax[1,2].plot(tri.x[i],tri.y[i],color = "forestgreen",marker = "o")
        ax[0,0].legend(loc = "upper left")

        max_error = np.amax([error_svd_hist,error_shred_hist])
        N_bins = 20
        bin_w = max_error / N_bins
        bin_edges = np.arange(0,N_bins) * bin_w
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        error_svd_hist_counts, _ = np.histogram(error_svd_hist, bins=bin_edges)
        error_shred_hist_counts, _ = np.histogram(error_shred_hist, bins=bin_edges)
        ax[1,0].set_title("max abs-error distribution over all predictions")
        ax[1,0].bar(bin_centers, error_svd_hist_counts, width=bin_w,alpha = 0.7, edgecolor="k",label = "svd-errors")
        ax[1,0].bar(bin_centers, error_shred_hist_counts, width=bin_w, alpha=0.7, edgecolor="k",label = "shred-errors")
        ax[1,0].legend()
        ax[1,0].set_ylabel("count")
        ax[1,0].set_xlabel("max abs error")

        buf = BytesIO()
        plt.savefig(buf, format = 'png')
        buf.seek(0)
        images.append(Image.open(buf))
        plt.close(fig)
        plt.show()
    
    save_path = folder_path + f"reconstruction_{nr_sensors}_sensors.gif"

    images[0].save(save_path,
               save_all=True,
               append_images=images[1:],
               duration=500,
               loop=0)
    
    
