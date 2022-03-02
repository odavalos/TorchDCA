import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .loss import *

class DCA(nn.Module):
    def __init__(self, latent_dim = 50, final_layer = 512, input_size = 13766):

        if input_size == None or latent_dim == None:
            raise ValueError('Please provide a value for each input_size, and latent_dim')
          
        super(DCA, self).__init__()
        self.in_dim = input_size;
        self.zdim = latent_dim;
        self.last_layer = final_layer;
        
        # Autoencoder
        self.encoder = Encoder(input_size = self.in_dim,  latent_dim = self.zdim); # encoder
        self.decoder = Decoder(input_size = self.in_dim, latent_dim = self.zdim); # decoder
        
        
        # mean
        self.mean = nn.Sequential(
                                nn.Linear(self.last_layer, self.in_dim),
                                MeanAct()
                                );
        # disp
        self.disp = nn.Sequential(
                                nn.Linear(self.last_layer, self.in_dim),
                                DispAct()
                                );
        # pi
        self.dropout_pi = nn.Sequential(
                                nn.Linear(self.last_layer, self.in_dim),
                                nn.Sigmoid()
                                );
        
        # reconstruction
        self.reconstruction = nn.linear(self.last_layer, self.in_dim)




    def forward(self, x):
        """
        Forward pass of the autoencoder
        """
        z = self.encoder(x);
        
        d = self.decoder(z);

        mean = self.mean(d);

        disp = self.disp(d);

        pi = self.dropout_pi(d);

        recon = self.reconstruction(d)


        return z, mean, disp, pi, recon
