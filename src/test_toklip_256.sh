export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CLIP_MODEL='original'
export TIMM_MODEL='toklip'
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=foobar
torchrun --nproc_per_node 1 --master_port 12152  -m open_clip_train.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --batch-size 128 \
    --epochs=32 \
    --workers=6 \
    --model ViT-SO400M-16-SigLIP2-256-toklip \
    --precision 'amp' \
    --seed 0 \
    --logs ./logs_eval/ \
    --report-to "tensorboard" \
    --imagenet-val IMAGENET_VAL \
    --coco-dir COCO_VAL_2014 \
    --coco-retrieval \
    --name 'test_256' \
    --pretrained 'TokLIP_S_256.pt' \