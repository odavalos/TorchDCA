import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from TorchDCA.utils import MeanAct, DispAct

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
        self.reconstruction = nn.Linear(self.last_layer, self.in_dim)




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
    
    
    def zinb_loss(x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        x = x.type(torch.FloatTensor)
        sf = scale_factor[:, None]
        mu = mean.type(torch.FloatTensor) * sf


        # negative binomial
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mu/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mu+eps)))
        final = t1 + t2

        # ZINB
        nb_case = final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mu+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda*torch.square(pi)
            result += ridge

        result = torch.mean(result)
        return result
