# import argparse
from pathlib import Path

from tap import Tap
from typing import Dict, List, Optional, Tuple, Union

try:
    from typing import Literal # typing.Literal is only available from Python 3.8 and up
except ImportError:
    from typing_extensions import Literal

class RTFMArgumentParser(Tap):
    # =============== 
    # network setting
    # ---------------
    backbone: Literal['i3d', 'c3d'] = 'i3d' # default backbone
    feature_size: int = 2048 # size of feature (default: 2048)
    attention_type: Literal['none', 'gate', 'base', 'both'] = 'none'
    gpus: List[int] = [0,1]
    lr: float = 0.001 # learning rates for steps
    scheduler: Optional[str] = None # scheduler for adjust lr
    batch_size: int = 32 # number of instances in a batch of data (default: 32)
    workers: int = 0 # number of workers in dataloader
    model_name: str = 'rtfm' # name to save model
    dataset: Literal['shanghaitech', 'ucf-crime', 'xd-violence'] = 'shanghaitech' # dataset to train
    plot_freq: int = 10 # frequency of plotting (default: 10)
    max_epoch: int = 15000 # maximum iteration to train (default: 15000)
    dropout: float = 0.7 # dropout ratio
    quantize_size: int = 32 # new temporal size for training

    # ============ 
    # path setting
    # ------------
    root_path: Path = 'data' # Directory path of data
    log_path: Path = 'logs' # Directory path of log
    checkpoint_path: Path = 'checkpoint' # Directory path of log
    resume: Optional[str] = None # trained checkpoint path

    # ========== 
    # evaluation
    # ----------
    evaluate_freq: int = 1 # frequency of running evaluation (default: 1)
    evaluate_min_step: int = 0 # min step of running evaluation (default: 5000)

    # ==== 
    # misc
    # ----
    viz: bool = True # viusualize mode
    seed: Optional[int] = -1 # random seed
    version: str = 'vad-1.0' # experiment version
    debug: bool = False # debug mode
    inference: bool = False # infernece mode
    div: bool = False # div_inference mode
    report_k: Optional[int] = None # maximum reported scores default 10
    descr: List[str] = ['RTFM', 'video', 'anomaly', 'detection'] # version description
