import torch
import torch.nn as nn

class zinb_loss(nn.Module):
    def __init__(self, x, mu, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        super(zinb_loss, self).__init__()
        self.input = x
        self.mu = mu
        self.theta = disp
        self.pi = pi
        self.sf = scale_factor
        self.rl = ridge_lambda

    def forward(self):
        eps = 1e-10
        x = self.input.type(torch.FloatTensor)
        mu = self.mu.type(torch.FloatTensor) * self.sf


        # negative binomial
        t1 = torch.lgamma(self.theta+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+self.theta+eps)
        t2 = (self.theta+x) * torch.log(1.0 + (mu/(self.theta+eps))) + (x * (torch.log(self.theta+eps) - torch.log(mu+eps)))
        final = t1 + t2

        # ZINB
        nb_case = final - torch.log(1.0-self.pi+eps)
        zero_nb = torch.pow(self.theta/(self.theta+mu+eps), self.theta)
        zero_case = -torch.log(self.pi + ((1.0-self.pi)*zero_nb)+eps)
        result = torch.where(x, 1e-8, zero_case, nb_case)


        ridge = self.ridge_lambda*torch.square(self.pi)
        result += ridge

        result = torch.mean(result)
        return result