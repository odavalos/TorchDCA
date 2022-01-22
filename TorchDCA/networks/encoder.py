import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim = 100, input_size = None):
        """
        
        The Encoder class
          
        """
        if latent_dim == None or input_size == None:
            raise ValueError('Must explicitly declare input size and latent space dimension')
            
        super(Encoder, self).__init__();
        self.in_dim = input_size;
        self.zdim = latent_dim;
        
        # feed forward layers  
        self.enc = nn.Sequential(
                                nn.Linear(self.inp_dim, 512),
                                nn.ReLU(),
                                nn.BatchNorm1d(512),
            
                                nn.Linear(512, self.zdim),
                                nn.ReLU(),
                                nn.BatchNorm1d(self.zdim)
                                           )
        
    def forward(self, x):        
        """
        
        Forward pass of the encoder
        
        """

        out = self.enc(x)   
        
        return out