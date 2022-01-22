import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim = 100, input_size = None):
        """
        
        The Decoder class
          
        """
        if latent_dim == None or input_size == None:
            raise ValueError('Must explicitly declare input size and latent space dimension')
            
        super(Decoder, self).__init__();
        self.in_dim = input_size;
        self.zdim = latent_dim;
        
        # feed forward layers  
        self.dec = nn.Sequential(
                                nn.Linear(self.zdim, 512),
                                nn.ReLU(),
                                nn.BatchNorm1d(512),
            
                                nn.Linear(512, self.inp_dim),
                                nn.ReLU(),
                                nn.BatchNorm1d(self.inp_dim)
                                           )
        
    def forward(self, z):        
        """
        
        Forward pass of the decoder
        
        """

        reconst = self.dec(z)   
        
        return reconst


