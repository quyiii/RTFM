CUDA_VISIBLE_DEVICES=2 python tools/trainval_anomaly_detector.py --dataset ucf-crime --version test --max_epoch 1000 --lr 0.001 --attention_type base --gpus 0 --batch_size 32