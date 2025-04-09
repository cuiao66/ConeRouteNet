GPU_NUM=4
export NVIDIA_VISIBLE_DEVICES=0,1,2,3
DATASET_ROOT='/root/mnt/DATASET/CARLA/Interfuser/'



python3 -m torch.distributed.launch --nproc_per_node=$GPU_NUM val.py $DATASET_ROOT  --dataset carla \
    --val-towns 5 --path_to_pth /root/DemoEnd2EndNet/interfuser.pth.tar \
    --val-weathers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 \
    --model interfuser_baseline --sched cosine --epochs 15 --warmup-epochs 5 --lr 0.0005 --batch-size 16  -j 16 --no-prefetcher --eval-metric l1_error \
    --opt adamw --opt-eps 1e-8 --weight-decay 0.05  \
    --scale 0.9 1.1 --saver-decreasing --clip-grad 10 --freeze-num -1 \
    --with-backbone-lr --backbone-lr 0.0002 \
    --multi-view --with-lidar --multi-view-input-size 3 128 128 \
    --experiment interfuser_baseline \
    --pretrained
