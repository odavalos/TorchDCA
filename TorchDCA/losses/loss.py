import torch
import torch.nn as nn


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



class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


