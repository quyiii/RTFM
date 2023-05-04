import torch.optim as optim
from .asam import SAM

def get_optimizer(optimizer_name, model, lr, weight_decay, rho):
    if 'adam' in optimizer_name:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if 'sam' in optimizer_name:
            optimizer = SAM(optimizer=optimizer, model=model, rho=rho)
    else:
        RuntimeError('optimizer {} error'.format(optimizer_name))
    return optimizer