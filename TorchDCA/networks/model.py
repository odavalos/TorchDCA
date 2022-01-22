import torch
import torch.nn as nn
from utils import *

class DCA(nn.Module):
    def __init__(self, input_size, latent_dim = 100, sigma=1.):

        if input_size == None or latent_dim == None:
            raise ValueError('Please provide a value for each input_size, and latent_dim')
          
        super(DCA, self).__init__()
        self.inp_dim = input_size
        self.zdim = latent_dim
        self.sigma = sigma

        # Autoencoder
        self.encoder = Encoder(input_size, latent_dim) # encoder
        self.decoder = Decoder(input_size, latent_dim) # decoder
        
        # mean
        self.decomean = nn.Sequential(
                                nn.Linear(self.zdim, 512),
                                MeanAct(),
            
                                nn.Linear(512, self.inp_dim),
                                MeanAct()
                                )
        # disp
        self.decodisp = nn.Sequential(
                                nn.Linear(self.zdim, 512),
                                DispAct(),

                                nn.Linear(512, self.inp_dim),
                                DispAct()
                                )
        # pi
        self.decopi = nn.Sequential(
                                nn.Linear(self.zdim, 512),
                                nn.Sigmoid(),

                                nn.Linear(512, self.inp_dim),
                                nn.Sigmoid()
                                )




    def forward(self, x):
        """

        Forward pass of the autoencoder

        """
        z = self.encoder(x)
        d = self.decoder(z)

        mean = self.decomean(z)
        disp = self.decodisp(z)
        pi = self.decopi(z)
        
        return z, d, mean, disp, pi