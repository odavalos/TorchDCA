import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim = 50, final_layer = 512, input_size = 13766):
        """

        The Decoder class

        """
        super(Decoder, self).__init__();
        self.in_dim = input_size;
        self.zdim = latent_dim;
        self.last_layer = final_layer;

        # decoder
        self.dec = nn.Sequential(
                                nn.Linear(self.zdim, 256),
                                nn.ReLU(),
                                nn.BatchNorm1d(256),
            
                                nn.Linear(256, 512),
                                nn.ReLU(),
                                nn.BatchNorm1d(512)
        )
        
            
            
    def forward(self, z):
            
            
        d = self.dec(z)



        return d

