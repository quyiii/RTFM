CUDA_VISIBLE_DEVICES=1 python tools/trainval_anomaly_detector.py --dataset ucf-crime --version rtfm-1000-cos-005 --max_epoch 1000 --scheduler cos --lr 0.005