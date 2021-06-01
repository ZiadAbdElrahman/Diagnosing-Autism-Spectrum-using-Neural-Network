

python train.py \
  --dataset_mode autism \
  --name dropout_facial_features \
  --model resnet \
  --netG resnet_6blocks \
  --gpu_ids 0 \
  --input_nc 3 \
  --batch_size 8 \
  --n_epochs 50 \
  --use_facial_features_data \
  --lr 2e-4 \
  --dropout_rate 0.5 \
  --ngf 32