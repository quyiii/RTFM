import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type('torch.cuda.FloatTensor')

from torch.nn import L1Loss, MSELoss, Sigmoid
from anomaly.losses import SigmoidMAELoss, sparsity_loss, smooth_loss, RTFM_loss, att_loss

def calculate_loss(regular_score, anomaly_score, regular_label,
anomaly_label, regular_crest, anomaly_crest, abn_scores, batch_size):
    loss_criterion = RTFM_loss(0.0001, 100)
    loss_magnitude = loss_criterion(
        regular_score,
        anomaly_score,
        regular_label,
        anomaly_label,
        regular_crest,
        anomaly_crest)

    loss_sparse = sparsity_loss(abn_scores, batch_size, 8e-3)
    loss_smooth = smooth_loss(abn_scores, 8e-4)

    cost = loss_magnitude + loss_smooth + loss_sparse
    return cost

def get_loss(model, input, regular_label, anomaly_label, batch_size):
    outputs = model(input)

    anomaly_score = outputs['anomaly_score']
    regular_score = outputs['regular_score']
    anomaly_crest = outputs['feature_select_anomaly']
    regular_crest = outputs['feature_select_regular']
    scores = outputs['scores']

    scores = scores.view(batch_size * 32 * 2, -1)

    scores = scores.squeeze()
    abn_scores = scores[batch_size * 32:]

    regular_label = regular_label[0:batch_size]
    anomaly_label = anomaly_label[0:batch_size]

    cost = calculate_loss(regular_score, anomaly_score, regular_label, anomaly_label, regular_crest, anomaly_crest, abn_scores, batch_size)
    
    return cost

def sam_step(optimizer, model, input, regular_label, anomaly_label, batch_size):
    loss = get_loss(model, input, regular_label, anomaly_label, batch_size)
    loss.backward()
    optimizer.ascent_step()

    loss = get_loss(model, input, regular_label, anomaly_label, batch_size)
    loss.backward()
    optimizer.descent_step()

    return loss.item()

def nor_step(optimizer, model, input, regular_label, anomaly_label, batch_size):
    loss = get_loss(model, input, regular_label, anomaly_label, batch_size)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def do_train(regular_loader, anomaly_loader, model, batch_size, optimizer, device, args):
    with torch.set_grad_enabled(True):
        model.train()

        """
        :param regular_video, anomaly_video
            - size: [bs, n=10, t=32, c=2048]
        """

        regular_video, regular_label = next(regular_loader)
        anomaly_video, anomaly_label = next(anomaly_loader)

        input = torch.cat((regular_video, anomaly_video), 0).cuda(device=args.gpus[0])

        if 'sam' in args.optimizer:
            loss = sam_step(optimizer, model, input, regular_label, anomaly_label, batch_size)
        else:
            loss = nor_step(optimizer, model, input, regular_label, anomaly_label, batch_size)

    return loss


