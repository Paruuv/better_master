import torch
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
from torch.utils.data import DataLoader
from ..objects.sindy import E_SINDy
from ..models.sequence_models.abstract_sequence import AbstractSequence
from ..models.decoder_models.abstract_decoder import AbstractDecoder
from ..models.latent_forecaster_models.abstract_latent_forecaster import AbstractLatentForecaster
from ..models.decoder_models.mlp_model import MLP
from ..models.decoder_models.unet_model import UNET
from ..models.sequence_models.gru_model import GRU
from ..models.sequence_models.lstm_model import LSTM
from ..models.sequence_models.transformer_model import TRANSFORMER
import warnings
from ..objects.dataset import TimeSeriesDataset
from ..models.latent_forecaster_models.sindy import SINDy_Forecaster
from ..models.latent_forecaster_models.lstm import LSTM_Forecaster
from ..objects.device import get_device

SEQUENCE_MODELS = {
    "LSTM": LSTM,
    "TRANSFORMER": TRANSFORMER,
    "GRU": GRU,
}

DECODER_MODELS = {
    "MLP": MLP,
    "UNET": UNET,
}

LATENT_FORECASTER_MODELS = {
    "SINDY_FORECASTER": SINDy_Forecaster,
    "LSTM_FORECASTER": LSTM_Forecaster
}

NUM_REPLICATES = 10

class PI_SHRED(torch.nn.Module):
    """
    SHallow REcurrent Decoder (SHRED) neural network architecture.

    SHRED leverages a sequence model to learn a latent representation of the temporal dynamics of sensor measurements, a 
    latent forecaster model to forecast the latent space into the future, and a decoder model to learn a mapping between 
    the latent space and the high-dimensional full-state space. The SHRED architecture enables accurate full-state 
    reconstructions and forecasts from limited sensors.

    Parameters
    ----------
    sequence_model : AbstractSequence or str, optional
        Sequence model instance (GRU, LSTM, Transformer) or its name.
        Default None → LSTM if not using SINDy_Forecaster, otherwise GRU.
    decoder_model : AbstractDecoder or str, optional
        Decoder model instance (MLP, UNET) or its name.
        Default None → MLP.
    latent_forecaster : AbstractLatentForecaster or str, optional
        Latent forecaster instance (SINDy_Forecaster, LSTM_Forecaster) or its name.
        Default None → no latent forecaster.

    Attributes
    ----------
    sequence : AbstractSequence
        The sequence model that encodes the temporal dynamics of sensor measurements in the latent space.
    decoder : AbstractDecoder
        The decoder model that maps the latent space back to the full‐state space.
    latent_forecaster : AbstractLatentForecaster or None
        The latent forecaster that forecasts future latent space states.
        If None, no latent forecaster is used.

    Examples
    --------
    >>> # basic SHRED
    >>> model = SHRED(sequence_model='LSTM', decoder_model='MLP', latent_forecaster='LSTM_Forecaster')
    >>> # SINDy SHRED
    >>> model = SHRED(sequence_model='GRU', decoder_model='MLP', latent_forecaster='SINDy_Forecaster')
    """
    def __init__(self, sequence_model: AbstractSequence = None, decoder_model: AbstractDecoder = None,
                 latent_forecaster: AbstractLatentForecaster = None):
        """
        Initialize a SHallow REcurrent Decoder (SHRED) model.

        Parameters
        ----------
        sequence_model : AbstractSequence or str, optional
            Sequence model instance (GRU, LSTM, Transformer) or its name.
            Default None → LSTM if not using SINDy_Forecaster, otherwise GRU.
        decoder_model : AbstractDecoder or str, optional
            Decoder model instance (MLP, UNET) or its name.
            Default None → MLP.
        latent_forecaster : AbstractLatentForecaster or str, optional
            Latent forecaster instance (SINDy_Forecaster, LSTM_Forecaster) or its name.
            Default None → no latent forecaster.

        Raises
        ------
        ValueError
            If a string name is given but not found in the corresponding model mapping.
        TypeError
            If an object of the wrong type is passed in for any of the three arguments.
        """
        super().__init__()
        if sequence_model is None:
            self.sequence = LSTM()
        elif isinstance(sequence_model, AbstractSequence):
            self.sequence = sequence_model
        elif isinstance(sequence_model, str):
            sequence_model = sequence_model.upper()
            if sequence_model not in SEQUENCE_MODELS:
                raise ValueError(f"Invalid sequence model: {sequence_model}. Choose from: {list(SEQUENCE_MODELS.keys())}")
            self.sequence = SEQUENCE_MODELS[sequence_model]()
        else:
            raise TypeError("Invalid type for 'sequence_model'. Must be str or an AbstractSequence instance or None.")

        if decoder_model is None:
            self.decoder = MLP()
        elif isinstance(decoder_model, AbstractDecoder):
            if latent_forecaster is not None and not isinstance(decoder_model, MLP):
                warnings.warn(
                    "`latent_forecaster` is not None, but `decoder_model` is not an instance of `MLP`. "
                    "The decoder is being overridden with `MLP()` for compatibility with the `latent_forecaster`.",
                    UserWarning
                )
                self.decoder = MLP()
            else:
                self.decoder = decoder_model
        elif isinstance(decoder_model, str):
            decoder_model = decoder_model.upper()
            if decoder_model not in DECODER_MODELS:
                raise ValueError(f"Invalid decoder model: {decoder_model}. Choose from: {list(DECODER_MODELS.keys())}")
            if latent_forecaster is not None and decoder_model!="MLP":
                warnings.warn(
                    "`latent_forecaster` is not None, but `decoder_model` is not set to \"MLP\". "
                    "The decoder is being overridden with `MLP()` for compatibility with the `latent_forecaster`.",
                    UserWarning
                )
                self.decoder = MLP()
            else:
                self.decoder = DECODER_MODELS[decoder_model]()
        else:
            raise TypeError("Invalid type for 'decoder'. Must be str or an AbstractDecoder instance or None.")
        self.num_replicates = NUM_REPLICATES
        self.latent_forecaster = None
        if latent_forecaster is not None:
            if isinstance(latent_forecaster, AbstractLatentForecaster):
                self.latent_forecaster = latent_forecaster
            elif isinstance(latent_forecaster, str):
                latent_forecaster = latent_forecaster.upper()
                if latent_forecaster not in LATENT_FORECASTER_MODELS:
                    raise ValueError(f"Invalid sequence model: {latent_forecaster}. Choose from: {list(LATENT_FORECASTER_MODELS.keys())}")
                self.latent_forecaster = LATENT_FORECASTER_MODELS[latent_forecaster]()
            else:
                raise TypeError("Invalid type for 'latent_forecaster'. Must be str or an AbstractLatentForecaster instance or None.")
            self.latent_forecaster.initialize(latent_dim = self.sequence.hidden_size)

    def forward(self, x, sindy=False):
        """
        Forward pass through the SHRED model.

        Parameters
        ----------
        x : torch.Tensor
            Input sensor sequences of shape (batch_size, lags, n_sensors).
        sindy : bool, optional
            Whether to compute SINDy regularization terms. Defaults to False.

        Returns
        -------
        torch.Tensor or tuple
            If sindy=False: Reconstructed full-state tensor.
            If sindy=True: Tuple of (reconstruction, target_latents, predicted_latents).
        """
        
        h_out = self.sequence(x)
        output = self.decoder(h_out)
        return output
    
    def _seq_model_outputs(self, x, sindy=False):
        """
        Get sequence model outputs with optional SINDy ensemble computation.

        Parameters
        ----------
        x : torch.Tensor
            Input sensor sequences.
        sindy : bool, optional
            Whether to compute SINDy ensemble outputs. Defaults to False.

        Returns
        -------
        torch.Tensor or tuple
            Sequence model outputs.
        """

        h_outs = self.sequence(x)
        return h_outs
    def compute_loss(x):
        r_ode = 0
        return r_ode


    # sindy_regularization previously set to 1.0
    def fit(self, train_dataset, val_dataset,  batch_size=64, num_epochs=200, lr=1e-3,
            optimizer="AdamW", verbose=True,plot = False,plot_modulo=50, threshold=0.5, base_threshold=0.0, patience=20,
            sindy_thres_epoch=20, weight_decay=0.01,model_dtype = torch.float32):
        """
        Train the SHRED model on the provided datasets.

        Parameters
        ----------
        train_dataset : TimeSeriesDataset
            Training dataset containing sensor sequences and target reconstructions.
        val_dataset : TimeSeriesDataset
            Validation dataset for monitoring training progress.
        batch_size : int, optional
            Batch size for training. Defaults to 64.
        num_epochs : int, optional
            Maximum number of training epochs. Defaults to 200.
        lr : float, optional
            Learning rate. Defaults to 1e-3.
        sindy_regularization : float, optional
            Weight for SINDy regularization term. Defaults to 0.
        optimizer : str, optional
            Optimizer type. Defaults to "AdamW".
        verbose : bool, optional
            Whether to print training progress. Defaults to True.
        threshold : float, optional
            SINDy thresholding value. Defaults to 0.5.
        base_threshold : float, optional
            Base threshold for SINDy. Defaults to 0.0.
        patience : int, optional
            Early stopping patience. Defaults to 20.
        sindy_thres_epoch : int, optional
            Frequency of SINDy thresholding. Defaults to 20.
        weight_decay : float, optional
            Weight decay for optimizer. Defaults to 0.01.

        Returns
        -------
        np.ndarray
            Array of validation errors for each epoch.
        """
        torch.set_default_dtype(model_dtype)
        device = get_device()
        input_size = train_dataset.X.shape[2] # nsensors + nparams
        output_size = train_dataset.Y.shape[1]
        lags = train_dataset.X.shape[1] # lags

        self.sequence.initialize(input_size=input_size, lags=lags, decoder_type=type(self.decoder).__name__,dtype = model_dtype)
        # self.sequence.to(device,dtype=torch.float64)
        
        self.decoder.initialize(input_size=self.sequence.output_size, output_size=output_size)
        # self.decoder.to(device,dtype=torch.float64)
        self.to(device,dtype = model_dtype)
        # shuffle is False (used to be True)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
        criterion = torch.nn.MSELoss()
        optimizer = PI_SHRED._get_optimizer(params=self.parameters(), optimizer=optimizer, lr=lr, weight_decay=weight_decay)
        val_error_list = []
        patience_counter = 0
        best_params = self.state_dict()
        best_val_error = float('inf')  # Initialize with a large value
        if plot:
            train_error = []
            val_error = []
            epochs = []

            layer_names = []
            for n, p in self.named_parameters():
                if (not p.requires_grad): 
                    continue
                if  "bias" in n: 
                    continue
                if "norms" in n:
                    continue
                layer_names.append(n)


            plt.ion()
            fig, ax = plt.subplots(1,2, figsize = (12,4))
            ax[0].set_title("Training & Testing Loss")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("MSE loss over epoch")
            ax[0].set_yscale("log")

            ax[1].set_title("Magnitude of model gradients")
            ax[1].set_xticks(range(len(layer_names)), [l.replace(".weight","") for l in layer_names], rotation=90)
            ax[1].set_ylabel("Magnitude of gradient")

            ax[1].set_yscale("log")
            (train_line,) = ax[0].plot([],[],label = "Train loss")
            (val_line,) = ax[0].plot([],[],label = "Test loss")
            ax[0].legend()
            (max_grad_line,) = ax[1].plot([],[],"o-",label = "max gradients")
            (mean_grad_line,) = ax[1].plot([],[],"o-",label = "mean gradients")
            ax[1].legend(loc = "lower left")
            fig.show()
            fig.canvas.draw()
            
        print("Fitting SHRED...")

        for epoch in range(1, num_epochs + 1):
            self.train()
            training_loss = 0.0
            val_loss_epoch = 0.0
            for train_data in train_loader:
                outputs = self.forward(train_data[0])
                optimizer.zero_grad()
                data_loss = criterion(outputs, train_data[1])
                loss = data_loss
                training_loss += loss.item()

                loss.backward()
                optimizer.step()
            # if verbose:
            #     print(f"Epoch {epoch}: Average training loss = {running_loss / len(train_loader):.6f}")

            # self.eval()
            
            for val_data in val_loader:
                outputs = self.forward(val_data[0])
                val_loss = criterion(outputs,val_data[1])
                val_loss_epoch += val_loss.item()
            
            if verbose: 
                print(f"Epoch {epoch}:Train loss: {training_loss / len(train_loader):.6f}, validation loss = {val_loss_epoch / len(val_loader):.6f}")
            
            if plot:
                if (epoch % plot_modulo) == 0:
                    epochs.append(epoch)
                    train_error.append(training_loss / len(train_loader))
                    val_error.append(val_loss_epoch / len(val_loader))
                    
                    ave_grads, max_grads = [], []
                    for n, p in self.named_parameters():
                        if (not p.requires_grad) or (p.grad is None): 
                            continue
                        if  "bias" in n: 
                            continue
                        if "norms" in n:
                            continue
                        g = p.grad.detach()
                        ave_grads.append(g.abs().mean().item())
                        max_grads.append(g.abs().max().item())
                    
                    train_line.set_data(epochs, train_error)
                    val_line.set_data(epochs, val_error)
                    # temp = np.arange(len(max_grads))
                    max_grad_line.set_data(np.arange(len(max_grads)),max_grads)
                    mean_grad_line.set_data(np.arange(len(max_grads)),ave_grads)
                    ax[1].relim()
                    ax[1].autoscale_view()

                    ax[0].relim()
                    ax[0].autoscale_view()
                    clear_output(wait=True)
                    display(fig)
                    plt.pause(0.001)
                    
            # if val_loss_epoch < best_val_error:
            #     best_val_error = val_loss_epoch
            #     best_params = self.state_dict()
            #     patience_counter = 0  # Reset if improvement
            # else:
            #     patience_counter += 1
            # if patience_counter >= patience:
            #     if verbose:
            #         print("Early stopping triggered: patience threshold reached.")
            #     break

        print('weee training done wewe')
        # self.load_state_dict(best_params)
        # device = next(self.parameters()).device

        return torch.tensor(val_error_list).detach().cpu().numpy()


    def evaluate(self, dataset: TimeSeriesDataset, batch_size: int=64):
        """
        Compute mean squared error on a held‐out test dataset.

        Parameters
        ----------
        test_dataset : torch.utils.data.Dataset
            Should return (X, Y) pairs just like your train/val datasets.
        batch_size : int, optional
            How many samples per batch. Defaults to 64.

        Returns
        -------
        float
            The MSE over all elements in the test set.
        """
        self.eval()
        loader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
        criterion = torch.nn.MSELoss(reduction="sum")
        device = next(self.parameters()).device
        total_loss = 0.0
        total_elements = 0
        with torch.no_grad():
            for X, Y in loader:
                X, Y = X.to(device), Y.to(device)
                preds = self(X) 
                # if sindy=True forward returns a tuple,
                # we only want the reconstruction
                if isinstance(preds, tuple):
                    preds = preds[0]
                loss = criterion(preds, Y)
                total_loss += loss.item()
                total_elements += Y.numel()
        # mean over every scalar element
        mse = total_loss / total_elements
        return mse


    @staticmethod
    def _get_optimizer(params, optimizer: str, lr: float, weight_decay: float):
        if optimizer == "AdamW":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif optimizer == "Adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer == "SGD":
            return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer == "RMSprop":
            return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
        elif optimizer == "Adagrad":
            return torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(
                f"Unsupported optimizer {optimizer!r}. Choose from: "
                "'Adam', 'AdamW', 'SGD', 'RMSprop', or 'Adagrad'."
            )