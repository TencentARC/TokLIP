
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from open_clip.factory import create_model_and_transforms


@dataclass
class ModelArgs:
    model: str = 'ViT-B-16'
    pretrained: str = None 
    precision: str = 'fp32'
    torchscript: bool = False
    force_quick_gelu: bool = False
    force_custom_text: bool = False
    force_patch_dropout: bool = None
    force_image_size: int = None
    image_mean: float = None
    image_std: float = None
    image_interpolation: str = None
    image_resize_mode: str = None
    aug_cfg: dict = field(default_factory=dict)
    pretrained_image: bool = False
    cache_dir: str = None
    init_visual_weights: bool = False
    init_logit_scale: bool = False


def create_toklip(model='ViT-SO400M-16-SigLIP2-384-toklip', model_path=None, image_size=384, device='cuda'):
    model_args = ModelArgs(model=model, pretrained=model_path, force_image_size=image_size)
    model_kwargs = {}
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_args.model,
        model_args.pretrained,
        precision=model_args.precision,
        device=device,
        jit=model_args.torchscript,
        force_quick_gelu=model_args.force_quick_gelu,
        force_custom_text=model_args.force_custom_text,
        force_patch_dropout=model_args.force_patch_dropout,
        force_image_size=model_args.force_image_size,
        image_mean=model_args.image_mean,
        image_std=model_args.image_std,
        image_interpolation=model_args.image_interpolation,
        image_resize_mode=model_args.image_resize_mode,  # only effective for inference
        aug_cfg=model_args.aug_cfg,
        pretrained_image=model_args.pretrained_image,
        output_dict=True,
        cache_dir=model_args.cache_dir,
        init_visual_weights=model_args.init_visual_weights,
        init_ori_logit_scale=model_args.init_logit_scale,
        **model_kwargs,
    )
    return model, preprocess_train, preprocess_val


def build_toklip_encoder(model, image_size, model_path):
    toklip_encoder, _, _ = create_toklip(model=model, image_size=image_size, model_path=model_path)
    toklip_encoder.eval()
    return toklip_encoder

if __name__ == '__main__':
    model = 'ViT-SO400M-16-SigLIP2-384-toklip'
    pretrained = 'TokLIP_L_384.pt'
    toklip_encoder = build_toklip_encoder(model, 384, pretrained)
    print(toklip_encoder.visual)


    model = 'ViT-SO400M-16-SigLIP2-256-toklip'
    pretrained = 'TokLIP_S_256.pt'
    toklip_encoder = build_toklip_encoder(model, 256, pretrained)
    print(toklip_encoder)