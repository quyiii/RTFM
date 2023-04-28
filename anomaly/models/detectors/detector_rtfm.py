import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init

from anomaly.models.modules import Attention, GatedAttention, TransMIL
torch.set_default_tensor_type('torch.FloatTensor')

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w) (B C T)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        # B x T x T
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        # B x T x C
        y = torch.matmul(f_div_C, g_x)
        # B x C x T 返回内存连续 有相同数据的tensor
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)



class Aggregate(nn.Module):
    # pyramid dilated for local
    # self-attention for global
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1,dilation=1, padding=1),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1,
                      stride=1, padding=0, bias = False),
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3,
                      stride=1, padding=1, bias=False), # should we keep the bias?
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            # nn.dropout(0.7)
        )

        self.non_local = NONLocalBlock1D(512, sub_sample=False, bn_layer=True)

    def forward(self, x):
            # x: (B, T, F)
            # out B x T x C
            out = x.permute(0, 2, 1)
            residual = out

            out1 = self.conv_1(out)
            out2 = self.conv_2(out)

            out3 = self.conv_3(out)
            out_d = torch.cat((out1, out2, out3), dim = 1)
            out = self.conv_4(out)
            out = self.non_local(out)
            out = torch.cat((out_d, out), dim=1)
            out = self.conv_5(out)   # fuse all the features together
            out = out + residual
            # B x T x C
            out = out.permute(0, 2, 1)

            return out

class RTFM(nn.Module):
    def __init__(self, attention_type, n_features, batch_size, quantize_size, dropout):
        super(RTFM, self).__init__()
        self.batch_size = batch_size
        self.quantize_size = quantize_size
        self.k_abn = self.quantize_size // 10
        self.k_nor = self.quantize_size // 10

        self.Aggregate = Aggregate(len_feature=2048)

        self.attention_type = attention_type
        if 'gate' in attention_type:
            self.attention = GatedAttention(n_features) 
        elif 'base' in attention_type or 'both' in attention_type:
            self.attention = Attention(n_features)
        else:
            self.attention = None

        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.drop_out = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.apply(weight_init)

    def get_topk_features_scores(self, features, scores, size):
        bs, ncrops, t, f = size

        # B x N = batch_size * 10
        normal_features = features[0:self.batch_size*ncrops]
        normal_scores = scores[0:self.batch_size]

        abnormal_features = features[self.batch_size*ncrops:]
        abnormal_scores = scores[self.batch_size:]

        # BN x T x C -> BN x T
        feat_magnitudes = torch.norm(features, p=2, dim=2)
        # BN x T -> B x N x T -> B x T
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)
        nfea_magnitudes = feat_magnitudes[0:self.batch_size]  # normal feature magnitudes
        afea_magnitudes = feat_magnitudes[self.batch_size:]  # abnormal feature magnitudes
        # print(feat_magnitudes.shape, self.batch_size, afea_magnitudes.shape)
        n_size = nfea_magnitudes.shape[0]

        if nfea_magnitudes.shape[0] == 1:  # this is for inference, the batch size is 1
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features

        select_idx = torch.ones_like(nfea_magnitudes)
        select_idx = self.drop_out(select_idx)

        #######  process abnormal videos -> select top3 feature magnitude  #######
        
        afea_magnitudes_drop = afea_magnitudes * select_idx
        # B x T -> B x k_abn
        idx_abn = torch.topk(afea_magnitudes_drop, self.k_abn, dim=1)[1]
        # B x k_abn -> B x k_abn x 1 -> B x k_abn x C
        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])

        abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
        abnormal_features = abnormal_features.permute(1, 0, 2, 3)

        total_select_abn_feature = torch.zeros(0, device=features.device)
        # -> BN x Topk x C
        for abnormal_feature in abnormal_features:
            # abnormal_feature B x T x C
            # idx_abn_feat B x k_abn x C

            # torch.gather 根据dim和index从tensor中取值
            # out[i][j][k] = input[i][index[i][j][k]][k] if dim == 1
            feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)   # top 3 features magnitude in abnormal bag
            
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
        # B x k_abn x 1 -> B x 1
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)  # top 3 scores in abnormal bag based on the top-3 magnitude


        ####### process normal videos -> select top3 feature magnitude #######

        select_idx_normal = torch.ones_like(nfea_magnitudes)
        select_idx_normal = self.drop_out(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        idx_normal = torch.topk(nfea_magnitudes_drop, self.k_nor, dim=1)[1]
        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

        normal_features = normal_features.view(n_size, ncrops, t, f)
        normal_features = normal_features.permute(1, 0, 2, 3)

        total_select_nor_feature = torch.zeros(0, device=features.device)
        for nor_fea in normal_features:
            feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)  # top 3 features magnitude in normal bag (hard negative)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

        idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1) # top 3 scores in normal bag

        return dict(
        anomaly_score = score_abnormal,
        regular_score = score_normal, 
        feature_select_anomaly = total_select_abn_feature,
        feature_select_regular = total_select_nor_feature,
        feature_magnitudes = feat_magnitudes
        )


    def rtfm_forward(self, inputs):
        # inputs B x T x N x C
        out = inputs
        bs, ncrops, t, f = out.size()

        # BN x T x C
        out = out.view(-1, t, f)

        # BN x T x C
        out = self.Aggregate(out)

        out = self.drop_out(out)

        features = out

        scores = self.relu(self.fc1(features))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))
        scores = scores.view(bs, ncrops, -1).mean(1)
        # B x T -> B x T x 1
        scores = scores.unsqueeze(dim=2)

        results = self.get_topk_features_scores(features, scores, (bs, ncrops, t, f))

        return dict(
        anomaly_score = results['anomaly_score'],
        regular_score = results['regular_score'], 
        feature_select_anomaly = results['feature_select_anomaly'],
        feature_select_regular = results['feature_select_regular'],
        scores = scores,
        feature_magnitudes = results['feature_magnitudes'])

    def get_mil_weights(self, features):
        # calculate score_wight by features
        # BN x T x C -> BN x T x 1
        weights = self.attention(features)
        # BN x T x 1 -> BN x T x 1
        weights = self.softmax(weights)

        return weights

    def get_weighted_features(self, features, weights, focus=True):
        if focus:
            # BN x 1 x C
            weighted_features = weights.permute(0, 2, 1) @ features
        else:
            # BN x T x C
            weighted_features = weights.expand_as(features) * features
        return weighted_features

    def get_score(self, features, ncrops=None):
        # BN x k x C
        t_scores = self.relu(self.fc1(features))
        t_scores = self.drop_out(t_scores)
        t_scores = self.relu(self.fc2(t_scores))
        t_scores = self.drop_out(t_scores)
        t_scores = self.sigmoid(self.fc3(t_scores))
        # BN x T x 1 -> B x T x 1
        if ncrops:
            bs = int(t_scores.shape[0] / ncrops)
            t_scores = t_scores.view(bs, ncrops, -1, 1).mean(1)
        # B x T x 1 or BN x T x 1
        return t_scores

    def att_forward(self, inputs):
        # inputs B x T x N x C

        out = inputs
        bs, ncrops, t, f = out.size()

        # BN x T x C
        out = out.view(-1, t, f)

        # BN x T x C
        out = self.Aggregate(out)

        out = self.drop_out(out)

        features = out

        # BN x T x C -> B x T x 1
        t_scores = self.get_score(features, ncrops)
        
        result = self.get_topk_features_scores(features, t_scores, (bs, ncrops, t, f))

        # BN x topk x C
        feature_select_anomaly = result['feature_select_anomaly']
        feature_select_regular = result['feature_select_regular']
        
        weights_select_abn = self.get_mil_weights(feature_select_anomaly)
        # BN x C
        features_select_abn = self.get_weighted_features(feature_select_anomaly, weights_select_abn)
        scores_select_abn = self.get_score(features_select_abn, ncrops).squeeze(dim=-1)

        weights_select_nor = self.get_mil_weights(feature_select_regular)
        features_select_nor = self.get_weighted_features(feature_select_regular, weights_select_nor)
        scores_select_nor = self.get_score(features_select_nor, ncrops).squeeze(dim=-1)

        return dict(
            anomaly_score = scores_select_abn,
            regular_score = scores_select_nor, 
            feature_select_anomaly = result['feature_select_anomaly'],
            feature_select_regular = result['feature_select_regular'],
            scores = t_scores,
            feature_magnitudes = result['feature_magnitudes']
        )

    def att_both_forward(self, inputs):
        # inputs B x T x N x C

        out = inputs
        bs, ncrops, t, f = out.size()

        # BN x T x C
        out = out.view(-1, t, f)

        # BN x T x C
        out = self.Aggregate(out)

        out = self.drop_out(out)

        features = out

        # BN x T x C -> B x T x 1
        t_scores = self.get_score(features, ncrops)
        
        result = self.get_topk_features_scores(features, t_scores, (bs, ncrops, t, f))

        # BN x topk x C
        feature_select_anomaly = result['feature_select_anomaly']
        feature_select_regular = result['feature_select_regular']
        
        weights_select_abn = self.get_mil_weights(feature_select_anomaly)
        # BN x C
        feature_focus_abn = self.get_weighted_features(feature_select_anomaly, weights_select_abn)
        scores_select_abn = self.get_score(feature_focus_abn, ncrops).squeeze(dim=-1)

        weights_select_nor = self.get_mil_weights(feature_select_regular)

        feature_focus_nor = self.get_weighted_features(feature_select_regular, weights_select_nor)
        scores_select_nor = self.get_score(feature_focus_nor, ncrops).squeeze(dim=-1)

        return dict(
            anomaly_score = scores_select_abn,
            regular_score = scores_select_nor, 
            feature_select_anomaly = feature_focus_abn.unsqueeze(dim=1),
            feature_select_regular = feature_focus_nor.unsqueeze(dim=1),
            scores = t_scores,
            feature_magnitudes = result['feature_magnitudes']
        )

    def forward(self, inputs):

        if 'none' in self.attention_type:
            result = self.rtfm_forward(inputs)
        elif 'both' in self.attention_type:
            result = self.att_both_forward(inputs)
        else:
            result = self.att_forward(inputs)
        return result