import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, in_channel=2048):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(nn.Linear(in_channel, in_channel // 4),
                                         nn.Tanh(),
                                         nn.Linear(in_channel // 4, 1))

    def forward(self, features):
        # BN x T x C -> BN x T x 1
        return self.attention(features)


class GatedAttention(nn.Module):
    def __init__(self, in_channel=2048):
        super(GatedAttention, self).__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(in_channel, in_channel // 4),
            nn.Tanh()
        )
        
        self.attention_U = nn.Sequential(
            nn.Linear(in_channel, in_channel // 4),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(in_channel // 4, 1)

    def forward(self, features):
        # BN x T x C -> BN x T x 1

        # BN x T x C -> BN x T x C/4
        at_v = self.attention_V(features)
        # BN x T x C -> BN x T x C/4
        at_u = self.attention_U(features)
        # BN x T x C/4 -> BN x T x 1
        return self.attention_weights(at_v * at_u)

