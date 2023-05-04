import torch
import torch.nn as nn
from .sigmoid_mae_loss import SigmoidMAELoss

class att_loss(nn.Module):
    def __init__(
            self,
            alpha,
            margin
        ):
        super(att_loss, self).__init__()
        
        self.alpha = alpha
        self.margin = margin
        self.sigmoid = nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        self.criterion = nn.BCELoss()
    
    def forward(
            self,
            regular_score, # (regular) video-level classification score of shape Bx1
            anomaly_score, # (anomaly) video-level classification score of shape Bx1
            regular_label, # (regular) video-level label of shape B
            anomaly_label, # (anomaly) video-level label of shape B
            regular_crest, # (regular) selected top snippet features of shape (Bxn)xtopkxC
            anomaly_crest, # (anomaly) selected top snippet features of shape (Bxn)xtopkxC
        ):

        label = torch.cat((regular_label, anomaly_label), 0)
        anomaly_score = anomaly_score
        regular_score = regular_score

        score = torch.cat((regular_score, anomaly_score), 0)
        score = score.squeeze()

        label = label.to(score.device)

        loss_cls = self.criterion(score, label)  # BCE loss in the score space

        loss_anomaly = torch.abs(
                self.margin - torch.norm(torch.mean(anomaly_crest, dim=1),
                p=2, dim=1)) # Bxn
        loss_regular = torch.norm(torch.mean(regular_crest, dim=1), p=2, dim=1) # Bxn
        loss = torch.mean((loss_anomaly + loss_regular) ** 2)

        loss_total = loss_cls + self.alpha * loss

        return loss_total