
import cv2
import torch
import joblib
import pickle
import numpy as np
import pandas as pd
import torch.utils.data as data

from tqdm import tqdm
from pathlib import Path
from utils import process_feat
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def softmax(scores, axis):
    es = np.exp(scores - scores.max(axis=axis)[..., None])
    return es / es.sum(axis=axis)[..., None]

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

class Dataset(data.Dataset):
    def __init__(
            self,
            data_root: Path = Path('data'),
            dataset: str='shanghaitech',
            backbone: str='i3d',
            quantize_size: int=32,
            is_normal: bool=True,
            transform = None,
            test_mode: bool=False,
            verbose: bool=False,
            data_file: str=None,
            ann_file: str=None,
        ):

        self.is_normal = is_normal
        self.dataset = dataset
        self.backbone = backbone
        self.quantize_size = quantize_size
        self.ann_file = ann_file

        self.subset = 'test' if test_mode else 'train'
        self.data_root = data_root
        self.data_dir = data_root.joinpath(dataset).joinpath(self.subset)

        # >> Load video list
        video_list = pd.read_csv(data_file)
        video_list = video_list['video-id'].values[:]

        self.transform = transform
        self.test_mode = test_mode

        self._prepare_data(video_list, verbose)

        self._prepare_classnames()

        self.num_frame = 0
        self.labels = None

        self.data_path_formatter = data_root.joinpath(
                dataset).joinpath(
                    backbone).joinpath(
                        self.subset).joinpath(
                            '{video_id}_{backbone}.npy')

    def _prepare_classnames(self):
        if 'shanghai' in self.dataset:
            self.classnames = []
        elif 'ucf' in self.dataset:
            self.classnames = ['Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
        elif 'xd' in self.dataset:
            self.classnames = []
        else:
            self.classnames = None

    def _prepare_frame_level_labels(self, video_list):
        import json
        ann_file = self.ann_file
        with open(ann_file, 'r') as fin:
            db = json.load(fin)

        ground_truths = list()
        for video_id in video_list:
            labels = db[video_id]['labels']
            ground_truths.append(labels)
        ground_truths = np.concatenate(ground_truths)
        return ground_truths

    def _prepare_data(self, video_list: list, verbose: bool=True):
        if self.test_mode is False:
            if 'shanghaitech' in self.dataset: index = 63
            elif 'ucf-crime' in self.dataset: index = 810

            self.video_list = video_list[index:] if self.is_normal else video_list[:index]
        else:
            self.video_list = video_list
            self.ground_truths = self._prepare_frame_level_labels(video_list)

        self.data_info = """
    Dataset description: [{state}] mode.

        - there are {vnum} videos in {dataset}.
        - subset: {subset}
        """.format(
            state = 'Regular' if self.is_normal else 'Anomaly',
            vnum = len(self.video_list),
            dataset = self.dataset,
            subset = self.subset.capitalize(),
        )

        if verbose: print(self.data_info)

    def __getitem__(self, index):

        video_id = self.video_list[index]

        data_path_formatter = str(self.data_path_formatter)

        label = self.get_label()  # get video level label 0/1

        data_path = data_path_formatter.format(
            video_id=video_id,
            backbone=self.backbone)

        features = np.load(data_path, allow_pickle=True) # tau x N x C, N=10
        features = np.array(features, dtype=np.float32) # tau x N x C

        if self.transform is not None:
            features = self.transform(features)


        if self.test_mode:
            return features
        else:
            t, n_group, channels = features.shape

            # quantize each video to 32-snippet-length video
            features = np.transpose(features, (1, 0, 2)) # N x T x C
            divided_features = []
            for feature in features:
                feature = self._process_feat(feature)
                divided_features.append(feature)
            # N x 32 x C
            features = np.array(divided_features, dtype=np.float32)

            return features, label

    def _process_feat(self, feat):
        # input: feat(T C)
        # output: new_feat(32 C)
        out_feat_len = self.quantize_size
        feat_len, feat_channel = feat.shape
        new_feat = np.zeros((out_feat_len, feat_channel)).astype(np.float32)

        # 均匀采点out_feat_len个
        r = np.linspace(0, feat_len, out_feat_len+1, dtype=np.int_)
        for i in range(out_feat_len):
            if r[i] != r[i+1]:
                # avg
                new_feat[i, :] = np.mean(feat[r[i]:r[i+1], :], 0)
            else:
                new_feat[i, :] = feat[r[i], :]
        return new_feat

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.video_list)

    def get_num_frames(self):
        return self.num_frame
