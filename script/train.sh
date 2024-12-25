python3 /home/ubuntu/train/train.py --arch archname --batch_size 32 --gpu '0' \
    --train_ps 64 --train_dir train_dir --env env \
    --val_dir valid_dir --save_dir save_dir \
    --dataset CT_training --warmup 