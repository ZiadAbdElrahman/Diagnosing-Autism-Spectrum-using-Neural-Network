

python train.py \
  --dataset_mode autism \
  --name inc_diff_arch_0.1 \
  --model resnet \
  --netG resnet_6blocks \
  --gpu_ids 0 \
  --input_nc 3 \
  --batch_size 32 \
  --n_epochs 50 \
  --use_facial_features_data \
  --lr 1e-4 \
  --dropout_rate 0.1 \
  --ngf 32