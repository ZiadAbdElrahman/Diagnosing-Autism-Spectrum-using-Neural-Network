
python train.py \
  --dataset_mode autism \
  --name new_arch_dropout_gray \
  --model resnet \
  --netG resnet_9blocks \
  --gpu_ids 0 \
  --input_nc 1 \
  --batch_size 8 \
  --n_epochs 30 \
  --lr 2e-4 \
  --dropout_rate 0.5 \
  --ngf 32