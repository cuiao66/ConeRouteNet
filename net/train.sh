GPU_NUM=3
export NVIDIA_VISIBLE_DEVICES=1,2,3
DATASET_ROOT='/root/mnt/DATASET/CARLA/Interfuser/'

# ./distributed_train.sh $GPU_NUM $DATASET_ROOT  --dataset carla_seq --seq_n 5 \
#     --train-towns 3 4  --val-towns 5 \
#     --train-weathers 0 1 2 3 4 5 6 7 8 9  --val-weathers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 \
#     --model DemoEnd2EndNet_baseline --sched cosine --epochs 15 --warmup-epochs 5 --lr 0.0005 --batch-size 16  -j 16 --no-prefetcher --eval-metric l1_error \
#     --opt adamw --opt-eps 1e-8 --weight-decay 0.05  \
#     --scale 0.9 1.1 --saver-decreasing --clip-grad 10 --freeze-num -1 \
#     --with-backbone-lr --backbone-lr 0.0002 \
#     --multi-view --with-lidar --multi-view-input-size 3 128 128 \
#     --experiment DemoEnd2EndNet_baseline \
#     --pretrained


./distributed_train.sh $GPU_NUM $DATASET_ROOT  --dataset carla --only_encoder \
    --train-towns 1 2 3 4 7 10  --val-towns 5 \
    --train-weathers 0 1 2 3 4 5 6 7  --val-weathers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 \
    --model DemoEnd2EndNet_baseline --sched cosine --epochs 35 --warmup-epochs 5 --lr 0.0005 --batch-size 16  -j 16 --no-prefetcher --eval-metric l1_error \
    --opt adamw --opt-eps 1e-8 --weight-decay 0.05  \
    --scale 0.9 1.1 --saver-decreasing --clip-grad 10 --freeze-num -1 \
    --experiment  train_with_traffic \
    --with-backbone-lr --backbone-lr 0.0002 \
    --multi-view  \
    # --pretrained \
    # --initial-checkpoint /root/DemoEnd2EndNet/net/output/AE_trafficPretrain/model_best.pth.tar \
    # --multi-view-input-size 3 128 128 \



# ./distributed_train.sh $GPU_NUM $DATASET_ROOT  --dataset carla --only_encoder\
#     --train-towns 1 2 3 4 6 7 10  --val-towns 5 \
#     --train-weathers 0 1 2 3 4 5 6 7 8 9 10 11 12  --val-weathers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 \
#     --model DemoEnd2EndNet_baseline --sched cosine --epochs 15 --warmup-epochs 5 --lr 0.0005 --batch-size 16  -j 16 --no-prefetcher --eval-metric l1_error \
#     --opt adamw --opt-eps 1e-8 --weight-decay 0.05  \
#     --scale 0.9 1.1 --saver-decreasing --clip-grad 10 --freeze-num -1 \
#     --with-backbone-lr --backbone-lr 0.0002 \
#     --multi-view --multi-view-input-size 3 128 128 \
#     --experiment  DemoEnd2EndNet_baseline \
#     --pretrained
