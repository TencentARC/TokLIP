export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CLIP_MODEL='original'
export TIMM_MODEL='toklip'
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=foobar
torchrun --nproc_per_node 1 --master_port 13215  -m open_clip_train.main \
    --batch-size 32 \
    --epochs=32 \
    --model ViT-SO400M-16-SigLIP2-384-toklip \
    --precision 'amp' \
    --seed 0 \
    --logs ./logs_eval/ \
    --report-to "tensorboard" \
    --imagenet-val IMAGENET_VAL \
    --coco-dir COCO_VAL_2014 \
    --coco-retrieval \
    --name 'test_384' \
    --pretrained 'TokLIP_L_384.pt' \