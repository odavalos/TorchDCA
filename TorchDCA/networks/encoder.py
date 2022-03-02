import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim = 50, input_size = 13766):
        """
        
        The Encoder class
          
        """
        if latent_dim == None or input_size == None:
            raise ValueError('Must explicitly declare input size and latent space dimension')
            
        super(Encoder, self).__init__();
        self.in_dim = input_size;
        self.zdim = latent_dim;

        self.enc = nn.Sequential(
                                nn.Linear(self.in_dim, 512),
                                nn.ReLU(),
                                nn.BatchNorm1d(512),
            
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.BatchNorm1d(256),
            
                                nn.Linear(256, self.zdim),
                                nn.ReLU(),
                                nn.BatchNorm1d(self.zdim)
                                           )
        
    def forward(self, x):        
        """
        
        Forward pass of the encoder
        
        """

        z = self.enc(x)
        
        
        return z
