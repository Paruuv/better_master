from .abstract_decoder import AbstractDecoder
import torch.nn as nn
import torch.nn.functional as F
import torch

class MLP_PI(AbstractDecoder):
    """
    Flexible Multilayer Perceptron (MLP) implementation.

    A fully connected decoder that maps a low-dimensional latent space
    back to a high-dimensional state with user-defined hidden layers.
    """

    def __init__(self, hidden_sizes=[350, 400], dropout=0.1):
        """
        Parameters:
        -----------
        hidden_sizes : list of int
            Sizes of each hidden layer in sequence. e.g. [256, 512]
        dropout : float
            Dropout probability applied after each hidden layer.
        """
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.dropout_prob = dropout
        self.layers = None
        self.norms = None
        self.dropout = nn.Dropout(self.dropout_prob)

    def initialize(self, input_size, output_size):
        """
        Initialize the MLP with input and output sizes.

        Parameters:
        -----------
        input_size : int
            Size of the input features.
        output_size : int
            Size of the output features.
        """
        super().initialize(input_size)
        print("yaaay we are initializing MLP_PI")
        # Build sequence of linear layers: input->hidden...->output
        sizes = [input_size] + self.hidden_sizes + [output_size]
        self.norms = nn.ModuleList()
        self.layers = nn.ModuleList()
        # self.layers = nn.ModuleList([
        #     nn.Linear(sizes[i], sizes[i+1])
        #     for i in range(len(sizes) - 1)
        # ])
        for i in range(len(sizes) - 1):
            layer = nn.Linear(sizes[i], sizes[i+1])
            
            # Kaiming init for ReLU
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            self.layers.append(layer)

            if i < len(sizes) - 2:
                norm = nn.LayerNorm(sizes[i+1])
                self.norms.append(norm)
                
    def forward(self, x):
        """
        Forward pass through MLP.

        Applies ReLU + dropout after each hidden layer; no activation on output.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation & dropout for all but the final layer
            if i < len(self.layers) - 1:
                x = self.norms[i](x)
                x = F.sigmoid(x)
                # x = torch.sin(x)
                x = self.dropout(x)
        return x

    @property
    def model_name(self):
        return "MLP_PI"

class MLP_TRUNK(AbstractDecoder):
    """
    Flexible Multilayer Perceptron (MLP) implementation.

    A fully connected decoder that maps a low-dimensional latent space
    back to a high-dimensional state with user-defined hidden layers.
    """

    def __init__(self, hidden_sizes=[350, 400], dropout=0.1):
        """
        Parameters:
        -----------
        hidden_sizes : list of int
            Sizes of each hidden layer in sequence. e.g. [256, 512]
        dropout : float
            Dropout probability applied after each hidden layer.
        """
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.dropout_prob = dropout
        self.layers = None
        self.norms = None
        self.dropout = nn.Dropout(self.dropout_prob)

    def initialize(self, input_size, output_size):
        """
        Initialize the MLP with input and output sizes.

        Parameters:
        -----------
        input_size : int
            Size of the input features.
        output_size : int
            Size of the output features.
        """
        super().initialize(input_size)
        print("yaaay we are initializing MLP_PI")
        # Build sequence of linear layers: input->hidden...->output
        sizes = [input_size] + self.hidden_sizes + [output_size]
        self.norms = nn.ModuleList()
        self.layers = nn.ModuleList()
        # self.layers = nn.ModuleList([
        #     nn.Linear(sizes[i], sizes[i+1])
        #     for i in range(len(sizes) - 1)
        # ])
        for i in range(len(sizes) - 1):
            layer = nn.Linear(sizes[i], sizes[i+1])
            
            # Kaiming init for ReLU
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            self.layers.append(layer)

            if i < len(sizes) - 2:
                norm = nn.LayerNorm(sizes[i+1])
                self.norms.append(norm)
                
    def forward(self, x):
        """
        Forward pass through MLP.

        Applies ReLU + dropout after each hidden layer; no activation on output.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation & dropout for all but the final layer
            if i < len(self.layers) - 1:
                x = self.norms[i](x)
                x = F.tanh(x)
                # x = torch.sin(x)
                x = self.dropout(x)
        return x

    @property
    def model_name(self):
        return "MLP_TRUNK"