CUDA_VISIBLE_DEVICES=2 python tools/trainval_anomaly_detector.py --dataset ucf-crime --version rtfm-500-cos-002 --max_epoch 500 --scheduler cos --lr 0.002