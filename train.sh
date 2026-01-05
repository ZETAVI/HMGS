exp_dir=./exp
config=configs/mipnerf360/bicycle.yaml
gpu=0
tag=ORIGINAL/bicycle

mkdir -p output/${tag}/logger

python launch.py \
    --exp_dir ${exp_dir} \
    --config ${config} \
    --gpu ${gpu} \
    --train \
    --eval \
    tag=${tag}\
    | tee output/${tag}/logger/training.log

