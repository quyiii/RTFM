import torch

def get_scheduler(optim, scheduler_name, max_epoch, min_lr=0, step_size=10, gamma=0.1):
    scheduler_name = scheduler_name.lower()
    if 'cos' in scheduler_name:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_epoch, eta_min=min_lr)
    elif 'step' in scheduler_name:
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=gamma)
    else:
        scheduler = None
    return scheduler