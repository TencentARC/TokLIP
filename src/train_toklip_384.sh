PROCESS=`ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9`    # kill all python processes

# For single-node training
if [ "$enable_single_node" = "1" ]; then
    export MASTER_ADDR=""
fi

if [ -z "$MASTER_ADDR" ]; then
    echo "MASTER_ADDR is not set, now set it to 127.0.0.1"
    export MASTER_ADDR="127.0.0.1"
    RANDOM_PORT=$(shuf -i 20000-29999 -n 1)
    export MASTER_PORT=$RANDOM_PORT
    export NODE_RANK="0"
    CUR_GPU_NUM=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    export NPROC_PER_NODE="$CUR_GPU_NUM"
    export NNODES="1"
    # export LOCAL_RANK="0"
fi

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CLIP_MODEL='original'
export TIMM_MODEL='toklip'
export LD_LIBRARY_PATH=foobar


echo "NODE_RANK: $NODE_RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "NNODES: $NNODES"


torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK \
-m  open_clip_train.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --train-data 'CC12M' \
    --train-num-samples 10968539 \
    --dataset-type webdataset \
    --lr "1e-5" \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 782 \
    --wd 0.1 \
    --batch-size 32 \
    --epochs=32 \
    --workers=6 \
    --model ViT-SO400M-16-SigLIP2-384-toklip \
    --precision 'amp' \
    --log-every-n-steps 32 \
    --seed 0 \
    --lock-text \
    --lock-text-freeze-layer-norm \
    --logs ./logs_toklip_384/ \
    --report-to "tensorboard" \
    --name 'cc12m-toklip-384' \
    --pretrained "./model/siglip2-so400m-vit-l16-384.pt" \
    --distill-model ViT-SO400M-16-SigLIP2-384 \
    --distill-pretrained "./model/siglip2-so400m-vit-l16-384.pt" \
    --image-resize-mode 'squash' \
    --image-interpolation 'bicubic' \
    --siglip-train \